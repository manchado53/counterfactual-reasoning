"""
Data structures for Gardner chess counterfactual analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ChessConsequenceRecord:
    """
    Record of a single state-action-consequence measurement for Gardner chess.

    One record is produced per white move in an episode. The consequence scores
    measure how different the game outcome distribution would have been if white
    had played an alternative legal move instead.

    Attributes:
        obs:                  (2875,) flat board observation from white's perspective
        action:               white's chosen move index (0–1224)
        timestep:             white's move number in the game (0-indexed)
        episode_return:       cumulative return accumulated before this move
        kl_score:             aggregated KL divergence across alternative moves
        kl_divergences:       KL divergence per alternative move index
        return_distributions: dict mapping move index -> (N,) array of rollout returns
        jsd_score:            aggregated Jensen-Shannon divergence
        jsd_divergences:      JSD per alternative move index
        tv_score:             aggregated Total Variation distance
        tv_distances:         TV per alternative move index
        wasserstein_score:    aggregated Wasserstein distance
        wasserstein_distances: Wasserstein per alternative move index
        pgx_state:            raw pgx.State (only stored if store_states=True)
    """
    # Core
    obs: np.ndarray
    action: int
    timestep: int
    episode_return: float = 0.0

    # Consequence scores
    kl_score: float = 0.0
    kl_divergences: Dict[int, float] = field(default_factory=dict)
    return_distributions: Dict[int, np.ndarray] = field(default_factory=dict)

    # Optional additional metrics
    jsd_score: Optional[float] = None
    jsd_divergences: Optional[Dict[int, float]] = None
    tv_score: Optional[float] = None
    tv_distances: Optional[Dict[int, float]] = None
    wasserstein_score: Optional[float] = None
    wasserstein_distances: Optional[Dict[int, float]] = None

    # Optional JAX state (for reproducibility / debugging)
    pgx_state: Optional[Any] = None

    def __repr__(self) -> str:
        metrics_str = f"KL={self.kl_score:.4f}"
        if self.jsd_score is not None:
            metrics_str += f", JSD={self.jsd_score:.4f}"
        if self.tv_score is not None:
            metrics_str += f", TV={self.tv_score:.4f}"
        if self.wasserstein_score is not None:
            metrics_str += f", W={self.wasserstein_score:.4f}"
        return (f"ChessConsequenceRecord(t={self.timestep}, "
                f"action={self.action}, {metrics_str})")

    def get_alternative_moves(self) -> List[int]:
        """Return list of alternative move indices that were evaluated."""
        return [m for m in self.return_distributions if m != self.action]

    def get_most_different_move(self) -> tuple:
        """Return (move_index, kl_divergence) of the most divergent alternative."""
        if not self.kl_divergences:
            return None, 0.0
        return max(self.kl_divergences.items(), key=lambda x: x[1])
