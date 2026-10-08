"""
ChessArrayReplayBuffer — array-backed PER buffer for Gardner chess.

Replaces PrioritizedReplayBuffer + ConsequenceReplayBuffer for the chess pipeline.
Pre-allocated numpy arrays + add_batch() via slice assignment = O(1) Python overhead
per chunk instead of O(65536) dict creations and list appends.

Supports:
  - Vanilla PER            (uniform=False, store_consequences=False)
  - Uniform sampling       (uniform=True)
  - Consequence-weighted PER (store_consequences=True)
      additive  Eq.4: p = mu * p_C + (1-mu) * p_delta
      multiplicative Eq.5: p = p_C^mu_c * p_delta^mu_delta / Z
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


class ChessArrayReplayBuffer:
    """
    Array-backed prioritized replay buffer for Gardner chess DQN.

    All transition fields are pre-allocated numpy arrays. add_batch() inserts
    n transitions via a single slice assignment — no Python loop, no dict creation.

    pgx.State leaves are stored in per-leaf pre-allocated arrays, initialized
    lazily on the first add_batch call that includes states_flat.

    sample() and sample_uniform() return a dict of numpy arrays (not list-of-dicts),
    so _update() and _score_buffer_transitions() use direct array indexing.
    """

    def __init__(
        self,
        capacity: int = 200_000,
        obs_dim: int = 2875,
        mask_dim: int = 1225,
        eps: float = 0.01,
        beta: float = 0.4,
        max_priority: float = 1.0,
        uniform: bool = False,
        store_consequences: bool = False,
        mu: float = 0.5,
        priority_mixing: str = 'additive',
        mu_c: float = 1.0,
        mu_delta: float = 1.0,
    ):
        self.capacity = capacity
        self.eps = eps
        self.beta = beta
        self.max_priority = max_priority
        self.uniform = uniform
        self.store_consequences = store_consequences
        self.mu = mu
        self.priority_mixing = priority_mixing
        self.mu_c = mu_c
        self.mu_delta = mu_delta

        # Pre-allocated transition arrays
        self.obs        = np.empty((capacity, obs_dim),     dtype=np.float32)
        self.next_obs   = np.empty((capacity, obs_dim),     dtype=np.float32)
        self.actions    = np.empty((capacity, 1),           dtype=np.int32)
        self.rewards    = np.empty(capacity,                dtype=np.float32)
        self.dones      = np.empty(capacity,                dtype=bool)
        self.masks      = np.empty((capacity, 1, mask_dim), dtype=bool)
        self.next_masks = np.empty((capacity, 1, mask_dim), dtype=bool)

        # Priority arrays (Equations 2-3)
        self.td_magnitudes      = np.zeros(capacity, dtype=np.float64)
        self.consequence_scores = np.zeros(capacity, dtype=np.float64)

        # pgx state storage — per-leaf pre-allocated arrays, initialized lazily
        self._state_leaves_np: Optional[List[np.ndarray]] = None
        self._state_treedef = None

        self._write_pos: int = 0
        self._size: int = 0
        self._cached_probs: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self._size

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    # ------------------------------------------------------------------ #
    # Write path                                                           #
    # ------------------------------------------------------------------ #

    def add_batch(
        self,
        obs:        np.ndarray,   # (n, obs_dim)
        next_obs:   np.ndarray,   # (n, obs_dim)
        actions:    np.ndarray,   # (n,) or (n, 1) int32
        rewards:    np.ndarray,   # (n,)
        dones:      np.ndarray,   # (n,) bool
        masks:      np.ndarray,   # (n, 1, mask_dim)
        next_masks: np.ndarray,   # (n, 1, mask_dim)
        states_flat = None,       # pytree with leaves (n, ...) — consequence DQN only
    ):
        """
        Store n transitions via numpy slice assignment — O(1) Python overhead.

        Circularly wraps when _write_pos + n > capacity.
        states_flat is only used when store_consequences=True.
        """
        n = len(obs)

        # Compute init priorities once for the whole batch (not per-transition)
        if self.store_consequences and self._size > 0:
            init_cs = float(np.mean(self.consequence_scores[:self._size]))
            init_td = float(np.max(self.td_magnitudes[:self._size]))
        else:
            init_cs = 0.0
            init_td = self.max_priority

        if states_flat is not None and self._state_leaves_np is None:
            self._init_state_storage(states_flat)

        end = self._write_pos + n

        if end <= self.capacity:
            sl = slice(self._write_pos, end)
            self._write_slice(sl, obs, next_obs, actions, rewards,
                              dones, masks, next_masks, init_cs, init_td)
            if states_flat is not None:
                self._assign_states(sl, states_flat, slice(None))
        else:
            first_n  = self.capacity - self._write_pos
            second_n = n - first_n
            sl1 = slice(self._write_pos, self.capacity)
            sl2 = slice(0, second_n)
            self._write_slice(sl1, obs[:first_n], next_obs[:first_n],
                              actions[:first_n], rewards[:first_n],
                              dones[:first_n], masks[:first_n],
                              next_masks[:first_n], init_cs, init_td)
            self._write_slice(sl2, obs[first_n:], next_obs[first_n:],
                              actions[first_n:], rewards[first_n:],
                              dones[first_n:], masks[first_n:],
                              next_masks[first_n:], init_cs, init_td)
            if states_flat is not None:
                self._assign_states(sl1, states_flat, slice(0, first_n))
                self._assign_states(sl2, states_flat, slice(first_n, n))

        self._write_pos = end % self.capacity
        self._size = min(self._size + n, self.capacity)
        self._cached_probs = None

    def _write_slice(self, sl, obs, next_obs, actions, rewards,
                     dones, masks, next_masks, init_cs, init_td):
        self.obs[sl]        = obs
        self.next_obs[sl]   = next_obs
        self.actions[sl]    = actions.reshape(-1, 1)
        self.rewards[sl]    = rewards
        self.dones[sl]      = dones
        self.masks[sl]      = masks
        self.next_masks[sl] = next_masks
        self.td_magnitudes[sl]      = init_td
        self.consequence_scores[sl] = init_cs

    # ------------------------------------------------------------------ #
    # pgx state storage                                                    #
    # ------------------------------------------------------------------ #

    def _init_state_storage(self, example_states_flat) -> None:
        """Allocate per-leaf arrays from the first batch's state pytree structure."""
        leaves, treedef = jax.tree_util.tree_flatten(example_states_flat)
        self._state_treedef = treedef
        # Each leaf shape: (n, *leaf_shape). Pre-allocate (capacity, *leaf_shape).
        self._state_leaves_np = [
            np.empty((self.capacity, *leaf.shape[1:]), dtype=leaf.dtype)
            for leaf in leaves
        ]

    def _assign_states(self, dst_sl: slice, states_flat, src_sl: slice) -> None:
        leaves, _ = jax.tree_util.tree_flatten(states_flat)
        for stored, leaf in zip(self._state_leaves_np, leaves):
            stored[dst_sl] = leaf[src_sl]

    def get_jax_state(self, idx: int) -> Any:
        """Reconstruct pgx.State at buffer slot idx as a JAX pytree."""
        if self._state_leaves_np is None:
            return None
        leaves = [jnp.array(stored[idx]) for stored in self._state_leaves_np]
        return jax.tree_util.tree_unflatten(self._state_treedef, leaves)

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def sample(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Sample batch using combined priorities (Equations 2-5).
        Returns (data_dict, indices, IS_weights) where data_dict values are arrays.
        """
        probs = self._compute_probs()
        indices = np.random.choice(self._size, size=batch_size, p=probs)
        weights = (np.ones(batch_size) if self.uniform
                   else 1.0 / (probs[indices] * self._size))
        return self._gather(indices), indices, weights

    def sample_uniform(self, n: int) -> Tuple[Dict, np.ndarray]:
        """Uniform sample for consequence scoring pass (Algorithm 2, line 11)."""
        n = min(n, self._size)
        indices = np.random.choice(self._size, size=n, replace=False)
        return self._gather(indices), indices

    def _gather(self, indices: np.ndarray) -> Dict:
        return {
            's':          self.obs[indices],
            "s'":         self.next_obs[indices],
            'a':          self.actions[indices],           # (B, 1) int32
            'r':          self.rewards[indices],
            'done':       self.dones[indices].astype(np.float32),
            'masks':      self.masks[indices],
            'next_masks': self.next_masks[indices],
        }

    # ------------------------------------------------------------------ #
    # Priority management                                                  #
    # ------------------------------------------------------------------ #

    def _compute_probs(self) -> np.ndarray:
        if self._cached_probs is not None:
            return self._cached_probs

        n = self._size
        td = np.nan_to_num(self.td_magnitudes[:n].copy())
        cs = np.nan_to_num(self.consequence_scores[:n].copy())

        if self.uniform:
            probs = np.ones(n) / n
        elif not self.store_consequences:
            # Vanilla PER: p^delta only (Equation 2)
            raw = (td + self.eps) ** self.beta
            s = raw.sum()
            probs = raw / s if s > 0 else np.ones(n) / n
        else:
            p_td_raw = (td + self.eps) ** self.beta
            s_td = p_td_raw.sum()
            p_td = p_td_raw / s_td if s_td > 0 else np.ones(n) / n

            p_c_raw = (cs + self.eps) ** self.beta
            s_c = p_c_raw.sum()
            p_c = p_c_raw / s_c if s_c > 0 else np.ones(n) / n

            if self.priority_mixing == 'multiplicative':
                combined = (p_c ** self.mu_c) * (p_td ** self.mu_delta)
            else:  # additive Eq.4
                combined = self.mu * p_c + (1.0 - self.mu) * p_td

            total = combined.sum()
            probs = combined / total if total > 0 else np.ones(n) / n

        self._cached_probs = probs
        return probs

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update |δ_j| for sampled transitions (Algorithm 2, line 16)."""
        self.td_magnitudes[indices] = np.abs(td_errors)
        self._cached_probs = None

    def update_consequence_scores(self, indices: np.ndarray, scores: np.ndarray):
        """Update m^C_j for scored transitions (Algorithm 2, line 12)."""
        self.consequence_scores[indices] = scores
        self._cached_probs = None
