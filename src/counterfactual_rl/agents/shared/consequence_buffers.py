"""Consequence-weighted Prioritized Experience Replay buffer (Algorithm 2, Equations 2-5)."""

import jax
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
        cce_balance: Optional[float] = None,
        target_ess_frac: Optional[float] = None,
        ess_recalib_every: int = 50,
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
        # ESS-matched mode: hold the sampler's effective sample size at
        # target_ess_frac and let cce_balance decide how much of that
        # concentration comes from the CCE score vs the TD error. Without this
        # the two ends of a balance sweep run at different concentrations --
        # measured 0.87 (pure TD) vs 0.47 (pure CCE) at a common exponent --
        # so a win could be sharpness rather than signal quality.
        self.cce_balance = cce_balance
        self.target_ess_frac = target_ess_frac
        self.ess_recalib_every = max(1, int(ess_recalib_every))
        self._ess_k = 1.0
        self._ess_calls = 0
        self._ess_k_saturated = False
        if target_ess_frac is not None:
            if cce_balance is None:
                raise ValueError("target_ess_frac requires cce_balance")
            if not 0.0 <= cce_balance <= 1.0:
                raise ValueError(f"cce_balance must be in [0,1], got {cce_balance}")
            if not 0.0 < target_ess_frac <= 1.0:
                raise ValueError(f"target_ess_frac must be in (0,1], got {target_ess_frac}")

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

    def add_batch(self, transitions: Dict, jax_states=None):
        """Add N transitions at once. transitions maps field names to 1-D arrays of length N.

        jax_states may be either a 1-D array (FrozenLake: a scalar int state per transition)
        or a *batched pytree* with leading axis N (JaxNav: a full ``State`` per transition).
        ``jax.tree.map(lambda leaf: leaf[i], ...)`` slices the i-th element for both, since a
        bare array is itself a single pytree leaf.
        """
        n = len(next(iter(transitions.values())))
        for i in range(n):
            t = {k: v[i] for k, v in transitions.items()}
            js = jax.tree.map(lambda leaf: leaf[i], jax_states) if jax_states is not None else None
            self.add(t, jax_state=js)

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

        if self.target_ess_frac is not None:
            # Eq 5 with the exponent scale solved to hit a target ESS.
            mu_c, mu_delta = self._solve_ess_exponents(p_c, p_td)
            combined = self._mix_log(np.log(np.maximum(p_c, 1e-300)),
                                     np.log(np.maximum(p_td, 1e-300)),
                                     mu_c, mu_delta)
        elif self.priority_mixing == 'multiplicative':
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

    @staticmethod
    def _mix_log(log_p_c, log_p_td, mu_c, mu_delta):
        """Normalised p^C^mu_c * p^delta^mu_delta, computed in log space.

        Done in logs because the direct product underflows hard: p ~ 1/N, so
        at N=1e5 and a combined exponent above ~60 every element rounds to 0,
        the sum is 0, and the caller silently falls back to uniform replay --
        a dead sampler that still logs the exponent it was asked for. The
        log-sum-exp form is exact at any exponent.
        """
        z = mu_c * log_p_c + mu_delta * log_p_td
        z -= z.max()
        c = np.exp(z)
        return c / c.sum()

    @classmethod
    def _ess_frac_at(cls, log_p_c, log_p_td, w, k):
        """ess/n of the mixed distribution at exponent scale k."""
        q = cls._mix_log(log_p_c, log_p_td, w * k, (1.0 - w) * k)
        return float(1.0 / np.sum(q ** 2) / len(q))

    def _solve_ess_exponents(self, p_c, p_td):
        """Bisect the exponent scale k so ess_frac hits target_ess_frac.

        ess_frac is monotonically decreasing in k (k=0 is uniform, ess_frac=1),
        so bisection is safe. Re-solved every ess_recalib_every calls rather
        than every call: the score histogram drifts slowly, and the solve costs
        ~40 vector passes over the buffer.

        If the target is unreachable even at K_MAX -- which happens when the
        signal is degenerate, e.g. every CCE score still 0 before the first
        scoring pass -- k saturates and _ess_k_saturated is set so the run can
        report that the requested concentration was never achieved.
        """
        K_MAX = 500.0
        self._ess_calls += 1
        if self._ess_calls % self.ess_recalib_every == 1:
            w = float(self.cce_balance)
            target = float(self.target_ess_frac)
            log_p_c = np.log(np.maximum(p_c, 1e-300))
            log_p_td = np.log(np.maximum(p_td, 1e-300))
            if self._ess_frac_at(log_p_c, log_p_td, w, K_MAX) > target:
                # even maximum sharpening cannot reach the target: the driving
                # signal is degenerate (e.g. every CCE score still 0 before the
                # first scoring pass). Clamp and flag rather than pretend.
                self._ess_k = K_MAX
                self._ess_k_saturated = True
            else:
                self._ess_k_saturated = False
                lo, hi = 0.0, K_MAX
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if self._ess_frac_at(log_p_c, log_p_td, w, mid) > target:
                        lo = mid
                    else:
                        hi = mid
                self._ess_k = 0.5 * (lo + hi)
        w = float(self.cce_balance)
        return w * self._ess_k, (1.0 - w) * self._ess_k

    def priority_diagnostics(self) -> dict:
        """Realized concentration of the sampling distribution.

        Read-only: uses the same cached priorities sample() draws from, so it
        reports what training actually did rather than a recomputation.

        ESS = 1 / sum(p^2) is the effective sample size of the priority
        distribution; ess_frac = ESS / n is 1.0 for uniform replay and falls
        toward 0 as replay concentrates on fewer transitions. The mu_c/mu_delta
        exponents set this only indirectly -- the same exponent gives a very
        different ess_frac depending on how sparse the score histogram is --
        so the sweep reports measured ess_frac, not the nominal exponents.
        """
        p = self._compute_priorities()
        n = len(p)
        if n == 0:
            return {}
        ess = float(1.0 / np.sum(p ** 2))
        order = np.sort(p)[::-1]
        cum = np.cumsum(order)
        valid = slice(None) if self._size == self.capacity else slice(self._size)
        cs = self.consequence_scores[valid]
        td = self.td_magnitudes[valid]
        return {
            'n': int(n),
            'ess': ess,
            'ess_frac': ess / n,
            # transitions supplying half of all draws, as a fraction of the buffer
            'top_half_frac': float((int(np.searchsorted(cum, 0.5)) + 1) / n),
            'p_max': float(order[0]),
            'uniform_fallback': bool(np.allclose(p, 1.0 / n)),
            'score_mean': float(cs.mean()),
            'score_std': float(cs.std()),
            'score_zero_frac': float(np.mean(cs <= 0.0)),
            'td_mean': float(td.mean()),
            'td_std': float(td.std()),
            'cce_balance': self.cce_balance,
            'target_ess_frac': self.target_ess_frac,
            'ess_k': float(self._ess_k) if self.target_ess_frac is not None else None,
            'ess_k_saturated': bool(self._ess_k_saturated),
        }

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
