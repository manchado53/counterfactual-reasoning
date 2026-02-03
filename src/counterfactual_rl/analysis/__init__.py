"""Counterfactual analysis and metrics

Import directly from submodules to avoid loading unnecessary dependencies:
  - counterfactual_rl.analysis.counterfactual (requires PyTorch/SB3)
  - counterfactual_rl.analysis.multidiscrete_counterfactual (requires PyTorch/SB3)
  - counterfactual_rl.analysis.smax_counterfactual (requires JAX)
  - counterfactual_rl.analysis.smax_vectorized_counterfactual (requires JAX)
  - counterfactual_rl.analysis.metrics (no special deps)
"""

# Only import dependency-free modules at package level
from counterfactual_rl.analysis.metrics import (
    compute_kl_divergence_kde,
    compute_wasserstein_distance,
    compute_total_variation,
    compute_jensen_shannon_divergence,
    compute_all_consequence_metrics,
)

__all__ = [
    "compute_kl_divergence_kde",
    "compute_wasserstein_distance",
    "compute_total_variation",
    "compute_jensen_shannon_divergence",
    "compute_all_consequence_metrics",
]

