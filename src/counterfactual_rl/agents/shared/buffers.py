"""Replay buffers for DQN training."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Samples transitions based on TD-error priorities, with importance
    sampling weights to correct for the non-uniform sampling.

    Internally uses a circular buffer with pre-allocated arrays for O(1) add/eviction.
    """

    def __init__(
        self,
        capacity: int = 100000,
        eps: float = 0.01,
        beta: float = 0.25,
        max_priority: float = 1.0,
        uniform: bool = False
    ):
        self.capacity = capacity
        self.eps = eps
        self.beta = beta
        self.max_priority = max_priority
        self.uniform = uniform

        # Circular buffer: pre-allocated to capacity, no shifting on eviction
        self.buffer: List[Any] = [None] * capacity
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float64)

        self._write_pos: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Dict):
        """Add a transition with maximum priority. O(1) — writes at _write_pos."""
        pos = self._write_pos
        self.buffer[pos] = transition
        self.priorities[pos] = (self.max_priority + self.eps) ** self.beta
        self._write_pos = (pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions based on priorities.

        Returns:
            Tuple of (transitions, indices, importance_sampling_weights)
        """
        if self._size < batch_size:
            raise ValueError(f"Not enough samples in buffer ({self._size} < {batch_size})")

        if self.uniform:
            indices = np.random.choice(self._size, size=batch_size)
            transitions = [self.buffer[idx] for idx in indices]
            weights = np.ones(batch_size)
            return transitions, indices, weights

        probs = self.priorities[:self._size].copy()
        probs /= probs.sum()
        indices = np.random.choice(self._size, size=batch_size, p=probs)

        transitions = [self.buffer[idx] for idx in indices]

        # Importance sampling weights: w_j = 1 / (p(j) * N)
        weights = 1.0 / (probs[indices] * self._size)

        return transitions, indices, weights

    def update_priority(self, index: int, td_error: float):
        """Update priority for a transition based on TD error."""
        if self.uniform:
            return
        self.priorities[index] = (abs(td_error) + self.eps) ** self.beta

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for multiple transitions."""
        if self.uniform:
            return
        for idx, td_error in zip(indices, td_errors):
            self.update_priority(idx, td_error)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self._size >= batch_size
