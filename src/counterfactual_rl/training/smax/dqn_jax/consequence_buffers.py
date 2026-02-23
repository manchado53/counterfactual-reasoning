"""Consequence-weighted Prioritized Experience Replay buffer (Algorithm 2, Equations 2-4)."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class ConsequenceReplayBuffer:
    """
    Prioritized replay buffer with consequence-weighted priorities.

    Combines TD-error priorities with consequence scores using mixing parameter mu:
        p(j) = mu * p^C(j) + (1-mu) * p^delta(j)     (Equation 4)

    where:
        p^delta(j) = (m^delta_j + eps)^beta / sum    (Equation 2)
        p^C(j)     = (m^C_j + eps)^beta / sum        (Equation 3)
    """

    def __init__(
        self,
        capacity: int = 100000,
        eps: float = 0.01,
        beta: float = 0.25,
        max_priority: float = 1.0,
        mu: float = 0.5,
    ):
        self.capacity = capacity
        self.eps = eps
        self.beta = beta
        self.max_priority = max_priority
        self.mu = mu

        self.buffer: List[Dict] = []
        self.consequence_scores: List[float] = []
        self.td_magnitudes: List[float] = []
        self.jax_states: List[Any] = []
        self.jax_obs: List[Any] = []

        self._cached_probs: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.buffer)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def add(self, transition: Dict, jax_state: Any = None, jax_obs: Any = None):
        """
        Add transition with initial priorities (Algorithm 2, lines 7-8).

        m^C_t  = mean(existing consequence_scores), or 0 if empty
        m^d_t  = max(existing td_magnitudes),       or max_priority if empty
        """
        # Initial consequence score: mean of existing (line 7)
        if self.consequence_scores:
            init_consequence = float(np.mean(self.consequence_scores))
        else:
            init_consequence = 0.0

        # Initial TD magnitude: max of existing (line 8)
        if self.td_magnitudes:
            init_td = float(np.max(self.td_magnitudes))
        else:
            init_td = self.max_priority

        self.buffer.append(transition)
        self.consequence_scores.append(init_consequence)
        self.td_magnitudes.append(init_td)
        self.jax_states.append(jax_state)
        self.jax_obs.append(jax_obs)

        # FIFO eviction if over capacity (line 9)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.consequence_scores.pop(0)
            self.td_magnitudes.pop(0)
            self.jax_states.pop(0)
            self.jax_obs.pop(0)

        self._cached_probs = None

    def _compute_priorities(self) -> np.ndarray:
        """Compute combined priorities (Equations 2-4)."""
        if self._cached_probs is not None:
            return self._cached_probs

        cs = np.array(self.consequence_scores, dtype=np.float64)
        td = np.array(self.td_magnitudes, dtype=np.float64)

        # Safety: replace any NaN/inf with 0 before priority computation
        cs = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)
        td = np.nan_to_num(td, nan=0.0, posinf=0.0, neginf=0.0)

        # Eq 3: p^C(j)
        p_c_raw = (cs + self.eps) ** self.beta
        p_c = p_c_raw / p_c_raw.sum()

        # Eq 2: p^delta(j)
        p_td_raw = (td + self.eps) ** self.beta
        p_td = p_td_raw / p_td_raw.sum()

        # Eq 4: combined
        combined = self.mu * p_c + (1.0 - self.mu) * p_td
        combined /= combined.sum()

        self._cached_probs = combined
        return combined

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample batch via combined priorities p(j) (line 13).

        Returns:
            (transitions, indices, importance_sampling_weights)
            IS weights: w_j = (p(j) * |D|)^{-1}  (line 14)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples ({len(self.buffer)} < {batch_size})")

        probs = self._compute_priorities()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)

        transitions = [self.buffer[idx] for idx in indices]

        # IS weights: w_j = 1 / (p(j) * N)  (line 14)
        N = len(self.buffer)
        weights = 1.0 / (probs[indices] * N)

        return transitions, indices, weights

    def sample_uniform(self, batch_size: int) -> Tuple[List[Dict], np.ndarray]:
        """Uniform sampling for consequence scoring pass (line 11)."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
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
