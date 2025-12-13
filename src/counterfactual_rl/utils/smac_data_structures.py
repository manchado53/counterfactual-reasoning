# File: /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/utils/smac_data_structures.py

"""
Data structures for SMAC multi-agent counterfactual analysis
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import numpy as np


@dataclass
class SmacConsequenceRecord:
    """
    Record of a single state-action-consequence measurement for SMAC environments.
    
    Designed specifically for multi-agent scenarios with continuous state observations
    and discrete joint actions.
    
    Attributes:
        state: Global state observation (np.ndarray) - full battlefield view
        action: Joint action tuple (e.g., (4, 1, 2) for 3 agents)
        timestep: Episode timestep when this decision was made
        episode_return: Total return accumulated so far in episode
        
        kl_score: Maximum KL divergence across alternative joint actions
        kl_divergences: Dict mapping alternative joint actions to KL divergence values
        return_distributions: Dict mapping joint actions to arrays of sampled returns
        
        # Battle statistics (optional)
        agents_alive: Number of allied agents still alive
        enemies_alive: Number of enemy agents still alive
        total_health: Total health of allied agents
        
        # Additional metrics (optional, computed by compute_all_metrics())
        jsd_score: Maximum Jensen-Shannon divergence
        jsd_divergences: Dict mapping alternative actions to JSD values
        tv_score: Maximum Total Variation distance
        tv_distances: Dict mapping alternative actions to TV values
        wasserstein_score: Maximum Wasserstein distance
        wasserstein_distances: Dict mapping alternative actions to Wasserstein values
    """
    # Core attributes
    state: np.ndarray
    action: Tuple[int, ...]  # Joint action for all agents
    timestep: int
    episode_return: float = 0.0
    
    # Consequence analysis
    kl_score: float = 0.0
    kl_divergences: Dict[Tuple[int, ...], float] = field(default_factory=dict)
    return_distributions: Dict[Tuple[int, ...], np.ndarray] = field(default_factory=dict)
    
    # Battle statistics
    agents_alive: Optional[int] = None
    enemies_alive: Optional[int] = None
    total_health: Optional[float] = None
    
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
        
        battle_info = ""
        if self.agents_alive is not None:
            battle_info = f", agents={self.agents_alive}, enemies={self.enemies_alive}"
        
        return (f"SmacConsequenceRecord(timestep={self.timestep}, "
                f"action={self.action}, {metrics_str}{battle_info})")
    
    def get_n_agents(self) -> int:
        """Get number of agents from action tuple."""
        return len(self.action)
    
    def format_action(self) -> str:
        """Format joint action as readable string."""
        return "(" + ",".join(str(a) for a in self.action) + ")"
    
    def get_alternative_actions(self) -> list:
        """Get list of alternative joint actions that were evaluated."""
        return list(self.return_distributions.keys())
    
    def get_most_different_action(self) -> Tuple[Tuple[int, ...], float]:
        """
        Get the alternative action most different from chosen action.
        
        Returns:
            Tuple of (alternative_action, kl_divergence)
        """
        if not self.kl_divergences:
            return None, 0.0
        
        max_action = max(self.kl_divergences.items(), key=lambda x: x[1])
        return max_action
    
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
            'mean_diff': returns1.mean() - returns2.mean(),
            'std_ratio': returns1.std() / (returns2.std() + 1e-8),
            'mean1': returns1.mean(),
            'std1': returns1.std(),
            'mean2': returns2.mean(),
            'std2': returns2.std(),
        }


@dataclass
class SmacEpisodeAnalysis:
    """
    Summary of counterfactual analysis for an entire SMAC episode.
    
    Aggregates records and provides episode-level statistics.
    """
    records: list[SmacConsequenceRecord]
    map_name: str
    episode_length: int
    episode_reward: float
    battle_won: bool
    
    def get_most_consequential_moments(self, top_n: int = 5) -> list[SmacConsequenceRecord]:
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
        freq = {}
        for record in self.records:
            freq[record.action] = freq.get(record.action, 0) + 1
        return freq
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the episode."""
        scores = np.array([r.kl_score for r in self.records])
        
        return {
            'mean_consequence': scores.mean(),
            'median_consequence': np.median(scores),
            'std_consequence': scores.std(),
            'max_consequence': scores.max(),
            'min_consequence': scores.min(),
            'n_timesteps': len(self.records),
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward,
            'battle_won': self.battle_won,
            'map_name': self.map_name,
        }
    
    def __repr__(self) -> str:
        stats = self.compute_statistics()
        return (f"SmacEpisodeAnalysis(map={self.map_name}, "
                f"timesteps={stats['n_timesteps']}, "
                f"mean_consequence={stats['mean_consequence']:.4f}, "
                f"battle_won={self.battle_won})")