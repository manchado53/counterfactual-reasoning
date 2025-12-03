"""
Data structures for consequential states analysis
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class ConsequenceRecord:
    """
    Record of a single state-action-consequence measurement.

    Attributes:
        state: State index in the environment
        action: Action taken by the agent
        position: Grid position (row, col) for spatial environments
        consequence_score: Maximum KL divergence across alternative actions (primary metric)
        kl_divergences: Dict mapping alternative actions to KL divergence values
        return_distributions: Dict mapping actions to arrays of sampled returns

        # Additional metrics (optional, computed by compute_all_metrics())
        jsd_score: Maximum Jensen-Shannon divergence across alternative actions
        jsd_divergences: Dict mapping alternative actions to JSD values
        tv_score: Maximum Total Variation distance across alternative actions
        tv_distances: Dict mapping alternative actions to TV values
        wasserstein_score: Maximum Wasserstein distance across alternative actions
        wasserstein_distances: Dict mapping alternative actions to Wasserstein values
    """
    state: int
    action: int
    position: Tuple[int, int]
    consequence_score: float
    kl_divergences: Dict[int, float]
    return_distributions: Dict[int, np.ndarray]

    # Optional additional metrics
    jsd_score: Optional[float] = None
    jsd_divergences: Optional[Dict[int, float]] = None
    tv_score: Optional[float] = None
    tv_distances: Optional[Dict[int, float]] = None
    wasserstein_score: Optional[float] = None
    wasserstein_distances: Optional[Dict[int, float]] = None

    def __repr__(self) -> str:
        metrics_str = f"KL={self.consequence_score:.4f}"
        if self.jsd_score is not None:
            metrics_str += f", JSD={self.jsd_score:.4f}"
        if self.tv_score is not None:
            metrics_str += f", TV={self.tv_score:.4f}"
        if self.wasserstein_score is not None:
            metrics_str += f", W={self.wasserstein_score:.4f}"

        return (f"ConsequenceRecord(state={self.state}, action={self.action}, "
                f"position={self.position}, {metrics_str})")
