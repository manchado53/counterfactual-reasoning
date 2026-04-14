"""Consequence-weighted Prioritized Experience Replay buffer (Algorithm 2, Equations 2-5)."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class ConsequenceReplayBuffer:
    """
    Prioritized replay buffer with consequence-weighted priorities.

    Supports two mixing modes:

    Additive (Eq 4):
        p(j) = mu * p^C(j) + (1-mu) * p^delta(j)

    Multiplicative (Eq 5):
        p(j) = p^C(j)^mu_C * p^delta(j)^mu_delta / Z

    where:
        p^delta(j) = (m^delta_j + eps)^beta / sum    (Equation 2)
        p^C(j)     = (m^C_j + eps)^beta / sum        (Equation 3)

    Internally uses a circular buffer with pre-allocated arrays for O(1) add/eviction.
    """

    def __init__(
        self,
        capacity: int = 100000,
        eps: float = 0.01,
        beta: float = 0.25,
        max_priority: float = 1.0,
        mu: float = 0.5,
        priority_mixing: str = 'additive',
        mu_c: float = 1.0,
        mu_delta: float = 1.0,
    ):
        if priority_mixing not in ('additive', 'multiplicative'):
            raise ValueError(
                f"priority_mixing must be 'additive' or 'multiplicative', got '{priority_mixing}'"
            )
        self.capacity = capacity
        self.eps = eps
        self.beta = beta
        self.max_priority = max_priority
        self.mu = mu
        self.priority_mixing = priority_mixing
        self.mu_c = mu_c
        self.mu_delta = mu_delta

        # Circular buffer: pre-allocated to capacity, no shifting on eviction
        self.buffer: List[Any] = [None] * capacity
        self.jax_states: List[Any] = [None] * capacity
        self.jax_obs: List[Any] = [None] * capacity
        self.consequence_scores: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.td_magnitudes: np.ndarray = np.zeros(capacity, dtype=np.float64)

        self._write_pos: int = 0   # next slot to write
        self._size: int = 0        # number of valid entries

        self._cached_probs: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self._size

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def add(self, transition: Dict, jax_state: Any = None, jax_obs: Any = None):
        """
        Add transition with initial priorities (Algorithm 2, lines 7-8).

        m^C_t  = mean(existing consequence_scores), or 0 if empty
        m^d_t  = max(existing td_magnitudes),       or max_priority if empty

        O(1): writes at _write_pos, advances pointer mod capacity.
        """
        if self._size > 0:
            valid = slice(None) if self._size == self.capacity else slice(self._size)
            init_consequence = float(np.mean(self.consequence_scores[valid]))
            init_td = float(np.max(self.td_magnitudes[valid]))
        else:
            init_consequence = 0.0
            init_td = self.max_priority

        pos = self._write_pos
        self.buffer[pos] = transition
        self.jax_states[pos] = jax_state
        self.jax_obs[pos] = jax_obs
        self.consequence_scores[pos] = init_consequence
        self.td_magnitudes[pos] = init_td

        self._write_pos = (pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        self._cached_probs = None

    def _compute_priorities(self) -> np.ndarray:
        """Compute combined priorities (Equations 2-5)."""
        if self._cached_probs is not None:
            return self._cached_probs

        valid = slice(None) if self._size == self.capacity else slice(self._size)
        cs = self.consequence_scores[valid].copy()
        td = self.td_magnitudes[valid].copy()

        # Safety: replace any NaN/inf with 0 before priority computation
        cs = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)
        td = np.nan_to_num(td, nan=0.0, posinf=0.0, neginf=0.0)

        # Eq 3: p^C(j)
        p_c_raw = (cs + self.eps) ** self.beta
        p_c = p_c_raw / p_c_raw.sum()

        # Eq 2: p^delta(j)
        p_td_raw = (td + self.eps) ** self.beta
        p_td = p_td_raw / p_td_raw.sum()

        if self.priority_mixing == 'multiplicative':
            # Eq 5: p(j) = p^C(j)^mu_C * p^delta(j)^mu_delta / Z
            combined = (p_c ** self.mu_c) * (p_td ** self.mu_delta)
        else:
            # Eq 4: p(j) = mu * p^C(j) + (1-mu) * p^delta(j)
            combined = self.mu * p_c + (1.0 - self.mu) * p_td

        # Underflow guard: fall back to uniform if all priorities collapse to 0
        total = combined.sum()
        if total == 0.0:
            combined = np.ones_like(combined) / len(combined)
        else:
            combined /= total

        self._cached_probs = combined
        return combined

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample batch via combined priorities p(j) (line 13).

        Returns:
            (transitions, indices, importance_sampling_weights)
            IS weights: w_j = (p(j) * |D|)^{-1}  (line 14)
        """
        if self._size < batch_size:
            raise ValueError(f"Not enough samples ({self._size} < {batch_size})")

        probs = self._compute_priorities()
        indices = np.random.choice(self._size, size=batch_size, p=probs)

        transitions = [self.buffer[idx] for idx in indices]

        # IS weights: w_j = 1 / (p(j) * N)  (line 14)
        N = self._size
        weights = 1.0 / (probs[indices] * N)

        return transitions, indices, weights

    def sample_uniform(self, batch_size: int) -> Tuple[List[Dict], np.ndarray]:
        """Uniform sampling for consequence scoring pass (line 11)."""
        if self._size < batch_size:
            batch_size = self._size

        indices = np.random.choice(self._size, size=batch_size, replace=False)
        transitions = [self.buffer[idx] for idx in indices]
        return transitions, indices

    def update_consequence_scores(self, indices: np.ndarray, scores: np.ndarray):
        """Update m^C_j for scored transitions (line 12)."""
        for idx, score in zip(indices, scores):
            self.consequence_scores[idx] = float(score)
        self._cached_probs = None

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update m^delta_j = |delta_j| (line 16)."""
        for idx, td_error in zip(indices, td_errors):
            self.td_magnitudes[idx] = float(abs(td_error))
        self._cached_probs = None

    def get_jax_state(self, index: int) -> Any:
        return self.jax_states[index]

    def get_jax_obs(self, index: int) -> Any:
        return self.jax_obs[index]
