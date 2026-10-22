"""Three replay rules with one storage and random-number path."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplaySample:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    weights: np.ndarray
    indices: np.ndarray
    probabilities: np.ndarray


class CartPoleReplayBuffer:
    """Uniform, canonical proportional PER, or the repository PER equation."""

    MODES = ("uniform", "paper_per", "repo_per")

    def __init__(self, capacity, observation_dim, mode, rng, *, paper_alpha=0.6,
                 paper_epsilon=1e-6, repo_exponent=0.25, repo_epsilon=0.01):
        if mode not in self.MODES:
            raise ValueError(f"Unsupported replay mode: {mode}")
        self.capacity = int(capacity)
        self.mode = mode
        self.rng = rng
        self.paper_alpha = float(paper_alpha)
        self.paper_epsilon = float(paper_epsilon)
        self.repo_exponent = float(repo_exponent)
        self.repo_epsilon = float(repo_epsilon)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.td_magnitudes = np.ones(capacity, dtype=np.float64)
        self.max_priority = 1.0
        self.position = 0
        self.size = 0

    def add(self, observation, action, reward, next_observation, done):
        i = self.position
        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_observation
        self.dones[i] = done
        self.td_magnitudes[i] = self.max_priority
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def probabilities(self):
        if self.size < 1:
            raise ValueError("Cannot sample an empty replay buffer")
        if self.mode == "uniform":
            return np.full(self.size, 1.0 / self.size, dtype=np.float64)
        td = self.td_magnitudes[:self.size]
        if self.mode == "paper_per":
            scores = np.power(td + self.paper_epsilon, self.paper_alpha)
        else:
            scores = np.power(td + self.repo_epsilon, self.repo_exponent)
        return scores / scores.sum()

    def sample(self, batch_size, *, beta):
        probabilities = self.probabilities()
        indices = self.rng.choice(self.size, size=batch_size, replace=True,
                                  p=None if self.mode == "uniform" else probabilities)
        selected = probabilities[indices]
        if self.mode == "uniform":
            weights = np.ones(batch_size, dtype=np.float32)
        elif self.mode == "paper_per":
            raw = np.power(self.size * selected, -beta)
            global_max = np.power(self.size * probabilities.min(), -beta)
            weights = (raw / global_max).astype(np.float32)
        else:
            weights = (1.0 / (self.size * selected)).astype(np.float32)
        return ReplaySample(
            self.observations[indices], self.actions[indices], self.rewards[indices],
            self.next_observations[indices], self.dones[indices], weights, indices,
            selected.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        magnitudes = np.abs(np.asarray(td_errors, dtype=np.float64))
        if not np.all(np.isfinite(magnitudes)):
            raise ValueError("Non-finite TD priority")
        for index, magnitude in zip(indices, magnitudes):
            self.td_magnitudes[index] = magnitude
        if magnitudes.size:
            self.max_priority = max(self.max_priority, float(magnitudes.max()))
