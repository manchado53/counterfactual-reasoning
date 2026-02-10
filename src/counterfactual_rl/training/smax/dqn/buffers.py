"""Replay buffers for DQN training."""

import numpy as np
from typing import Dict, List, Tuple


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Samples transitions based on TD-error priorities, with importance
    sampling weights to correct for the non-uniform sampling.
    """

    def __init__(
        self,
        capacity: int = 100000,
        eps: float = 0.01,
        beta: float = 0.25,
        max_priority: float = 1.0
    ):
        """
        Args:
            capacity: Maximum buffer size
            eps: Small constant added to priorities to ensure non-zero sampling
            beta: Exponent for priority-based sampling (0 = uniform, 1 = full prioritization)
            max_priority: Initial priority for new transitions
        """
        self.capacity = capacity
        self.eps = eps
        self.beta = beta
        self.max_priority = max_priority

        self.buffer: List[Dict] = []
        self.priorities: List[float] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Dict):
        """
        Add a transition with maximum priority.

        Args:
            transition: Dict with keys like 's', 'a', 'r', "s'", 'done', etc.
        """
        self.buffer.append(transition)
        priority = (self.max_priority + self.eps) ** self.beta
        self.priorities.append(priority)

        # Remove oldest if over capacity
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions based on priorities.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (transitions, indices, importance_sampling_weights)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer ({len(self.buffer)} < {batch_size})")

        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)

        transitions = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        prob_uniform = 1.0 / len(self.buffer)
        weights = np.array([prob_uniform / probs[idx] for idx in indices])

        return transitions, indices, weights

    def update_priority(self, index: int, td_error: float):
        """
        Update priority for a transition based on TD error.

        Args:
            index: Index of transition in buffer
            td_error: Temporal difference error
        """
        priority = (abs(td_error) + self.eps) ** self.beta
        self.priorities[index] = priority

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for multiple transitions.

        Args:
            indices: Array of indices
            td_errors: Array of TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            self.update_priority(idx, td_error)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size
