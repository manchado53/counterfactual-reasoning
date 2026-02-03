"""
Data structures for SMAX (JaxMARL) multi-agent counterfactual analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List
import numpy as np


@dataclass
class SMAXConsequenceRecord:
    """
    Record of a single state-action-consequence measurement for SMAX environments.

    Similar to SmacConsequenceRecord but adapted for JAX/SMAX specifics:
    - Observations stored as dict per agent (not flattened)
    - Optional JAX state storage for reproducibility
    - Optional RNG key storage

    Attributes:
        obs: Dictionary of observations per agent
        action: Joint action tuple (e.g., (4, 1, 2) for 3 agents)
        timestep: Episode timestep when this decision was made
        episode_return: Total return accumulated so far in episode

        state: Optional JAX state pytree (for reproducibility/debugging)
        rng_key: Optional JAX PRNGKey at this timestep

        kl_score: Maximum KL divergence across alternative joint actions
        kl_divergences: Dict mapping alternative joint actions to KL divergence values
        return_distributions: Dict mapping joint actions to arrays of sampled returns

        # Additional metrics (optional, computed by compute_all_metrics())
        jsd_score: Maximum Jensen-Shannon divergence
        jsd_divergences: Dict mapping alternative actions to JSD values
        tv_score: Maximum Total Variation distance
        tv_distances: Dict mapping alternative actions to TV values
        wasserstein_score: Maximum Wasserstein distance
        wasserstein_distances: Dict mapping alternative actions to Wasserstein values
    """
    # Core attributes
    obs: Dict[str, np.ndarray]  # Observations per agent
    action: Tuple[int, ...]      # Joint action for all agents
    timestep: int
    episode_return: float = 0.0

    # Optional JAX state storage (for reproducibility)
    state: Optional[Any] = None
    rng_key: Optional[Any] = None

    # Consequence analysis
    kl_score: float = 0.0
    kl_divergences: Dict[Tuple[int, ...], float] = field(default_factory=dict)
    return_distributions: Dict[Tuple[int, ...], np.ndarray] = field(default_factory=dict)

    # Optional additional metrics
    jsd_score: Optional[float] = None
    jsd_divergences: Optional[Dict[Tuple[int, ...], float]] = None
    tv_score: Optional[float] = None
    tv_distances: Optional[Dict[Tuple[int, ...], float]] = None
    wasserstein_score: Optional[float] = None
    wasserstein_distances: Optional[Dict[Tuple[int, ...], float]] = None

    def __repr__(self) -> str:
        metrics_str = f"KL={self.kl_score:.4f}"
        if self.jsd_score is not None:
            metrics_str += f", JSD={self.jsd_score:.4f}"
        if self.tv_score is not None:
            metrics_str += f", TV={self.tv_score:.4f}"
        if self.wasserstein_score is not None:
            metrics_str += f", W={self.wasserstein_score:.4f}"

        return (f"SMAXConsequenceRecord(timestep={self.timestep}, "
                f"action={self.action}, {metrics_str})")

    def get_n_agents(self) -> int:
        """Get number of agents from action tuple."""
        return len(self.action)

    def format_action(self) -> str:
        """Format joint action as readable string."""
        return "(" + ",".join(str(a) for a in self.action) + ")"

    def get_alternative_actions(self) -> List[Tuple[int, ...]]:
        """Get list of alternative joint actions that were evaluated."""
        return list(self.return_distributions.keys())

    def get_most_different_action(self) -> Tuple[Optional[Tuple[int, ...]], float]:
        """
        Get the alternative action most different from chosen action.

        Returns:
            Tuple of (alternative_action, kl_divergence)
        """
        if not self.kl_divergences:
            return None, 0.0

        max_action, max_kl = max(self.kl_divergences.items(), key=lambda x: x[1])
        return max_action, max_kl

    def compare_returns(self, action1: Tuple[int, ...], action2: Tuple[int, ...]) -> Dict[str, float]:
        """
        Compare return statistics between two actions.

        Args:
            action1: First joint action
            action2: Second joint action

        Returns:
            Dict with statistical comparison
        """
        if action1 not in self.return_distributions or action2 not in self.return_distributions:
            return {}

        returns1 = self.return_distributions[action1]
        returns2 = self.return_distributions[action2]

        return {
            'mean_diff': float(returns1.mean() - returns2.mean()),
            'std_ratio': float(returns1.std() / (returns2.std() + 1e-8)),
            'mean1': float(returns1.mean()),
            'std1': float(returns1.std()),
            'mean2': float(returns2.mean()),
            'std2': float(returns2.std()),
        }


@dataclass
class SMAXEpisodeAnalysis:
    """
    Summary of counterfactual analysis for an entire SMAX episode.

    Aggregates records and provides episode-level statistics.
    """
    records: List[SMAXConsequenceRecord]
    scenario_name: str
    episode_length: int
    episode_reward: float
    battle_won: bool

    def get_most_consequential_moments(self, top_n: int = 5) -> List[SMAXConsequenceRecord]:
        """Get the top-N most consequential moments."""
        sorted_records = sorted(self.records, key=lambda r: r.kl_score, reverse=True)
        return sorted_records[:top_n]

    def get_consequence_over_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get consequence scores over episode timesteps."""
        timesteps = np.array([r.timestep for r in self.records])
        scores = np.array([r.kl_score for r in self.records])
        return timesteps, scores

    def get_action_frequencies(self) -> Dict[Tuple[int, ...], int]:
        """Count how often each joint action was taken."""
        freq: Dict[Tuple[int, ...], int] = {}
        for record in self.records:
            freq[record.action] = freq.get(record.action, 0) + 1
        return freq

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the episode."""
        if not self.records:
            return {
                'mean_consequence': 0.0,
                'median_consequence': 0.0,
                'std_consequence': 0.0,
                'max_consequence': 0.0,
                'min_consequence': 0.0,
                'n_timesteps': 0,
                'episode_length': self.episode_length,
                'episode_reward': self.episode_reward,
                'battle_won': self.battle_won,
                'scenario_name': self.scenario_name,
            }

        scores = np.array([r.kl_score for r in self.records])

        return {
            'mean_consequence': float(scores.mean()),
            'median_consequence': float(np.median(scores)),
            'std_consequence': float(scores.std()),
            'max_consequence': float(scores.max()),
            'min_consequence': float(scores.min()),
            'n_timesteps': len(self.records),
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward,
            'battle_won': self.battle_won,
            'scenario_name': self.scenario_name,
        }

    def __repr__(self) -> str:
        stats = self.compute_statistics()
        return (f"SMAXEpisodeAnalysis(scenario={self.scenario_name}, "
                f"timesteps={stats['n_timesteps']}, "
                f"mean_consequence={stats['mean_consequence']:.4f}, "
                f"battle_won={self.battle_won})")
