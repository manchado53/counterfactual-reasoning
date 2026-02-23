"""
Metrics for comparing return distributions
"""

import numpy as np
from scipy.stats import gaussian_kde
from typing import Optional
import warnings


def compute_kl_divergence_kde(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    n_points: int = 100,
    bandwidth: Optional[str] = 'scott'
) -> float:
    """
    Compute KL divergence between two distributions using Kernel Density Estimation.

    KL(P||Q) = ∑ P(x) log(P(x)/Q(x))

    We use Gaussian KDE to estimate the densities from samples, then
    discretize and compute KL divergence.
    
    Note: KL divergence is unbounded and can be arbitrarily large when
    distributions differ significantly. For constant distributions with
    different values, returns float('inf') representing mathematical infinity.

    Args:
        samples_p: Samples from distribution P
        samples_q: Samples from distribution Q
        n_points: Number of points for discretization
        bandwidth: Bandwidth method for KDE ('scott' or 'silverman')

    Returns:
        KL divergence value (nats), unbounded. Returns 0.0 for edge cases.

    Example:
        >>> p_samples = np.random.normal(0, 1, 1000)
        >>> q_samples = np.random.normal(1, 1, 1000)
        >>> kl = compute_kl_divergence_kde(p_samples, q_samples)
        >>> print(f"KL divergence: {kl:.4f}")
    """
    # Handle edge cases
    if len(samples_p) < 2 or len(samples_q) < 2:
        warnings.warn("Insufficient samples for KDE. Returning 0.")
        return 0.0

    # If distributions are identical, return 0
    if np.allclose(samples_p, samples_q):
        return 0.0

    # Check for zero variance (constant distributions)
    var_p = np.var(samples_p)
    var_q = np.var(samples_q)

    # If both are constant
    if var_p < 1e-10 and var_q < 1e-10:
        # Both constant - check if same value
        if np.allclose(samples_p.mean(), samples_q.mean()):
            return 0.0
        else:
            # Different constant values - KL divergence is mathematically infinite
            return float('inf')

    # If one is constant and the other isn't
    if var_p < 1e-10 or var_q < 1e-10:
        # Use simpler metric: compare mean differences
        # Scale by the variance of the non-constant distribution
        mean_diff = abs(samples_p.mean() - samples_q.mean())
        active_var = max(var_p, var_q)
        if active_var > 0:
            # Normalized squared difference (like a Z-score squared)
            return (mean_diff ** 2) / active_var
        else:
            return 0.0

    try:
        # Determine common support range
        all_samples = np.concatenate([samples_p, samples_q])
        x_min, x_max = all_samples.min(), all_samples.max()

        # Add small margin to avoid boundary issues
        margin = 0.1 * (x_max - x_min) if x_max > x_min else 0.1
        x_min -= margin
        x_max += margin

        # Create evaluation points
        x_eval = np.linspace(x_min, x_max, n_points)

        # Estimate densities using KDE
        kde_p = gaussian_kde(samples_p, bw_method=bandwidth)
        kde_q = gaussian_kde(samples_q, bw_method=bandwidth)

        # Evaluate densities
        p_density = kde_p(x_eval)
        q_density = kde_q(x_eval)

        # Normalize to ensure they sum to 1 (important for discrete KL)
        p_density = p_density / np.sum(p_density)
        q_density = q_density / np.sum(q_density)

        # Add epsilon to avoid log(0) and division by zero
        epsilon = 1e-6  # Larger epsilon for numerical stability with discrete distributions
        p_density = np.clip(p_density, epsilon, None)
        q_density = np.clip(q_density, epsilon, None)

        # CRITICAL: Re-normalize after clipping to maintain probability distribution property
        # Without this, clipping breaks the sum-to-1 assumption and causes extremely high KL values
        p_density = p_density / np.sum(p_density)
        q_density = q_density / np.sum(q_density)

        # Compute log ratio with safety bounds to prevent extreme values
        log_ratio = np.log(p_density / q_density)
        log_ratio = np.clip(log_ratio, -20, 20)  # Prevents numerical overflow

        # Compute KL divergence
        kl_div = np.sum(p_density * log_ratio)

        return max(0.0, kl_div)  # KL should be non-negative

    except Exception as e:
        warnings.warn(f"KL divergence computation failed: {e}. Returning 0.")
        return 0.0


def compute_wasserstein_distance(
    samples_p: np.ndarray,
    samples_q: np.ndarray
) -> float:
    """
    Compute Wasserstein (Earth Mover's) distance between two distributions.

    This is an alternative metric to KL divergence that is symmetric and
    has geometric interpretation.

    Args:
        samples_p: Samples from distribution P
        samples_q: Samples from distribution Q

    Returns:
        Wasserstein distance (1-Wasserstein, or Earth Mover's Distance)
    """
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(samples_p, samples_q)


def compute_total_variation(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    n_bins: int = 50
) -> float:
    """
    Compute Total Variation distance between two distributions.

    TV(P, Q) = 0.5 * ∑ |P(x) - Q(x)|

    Args:
        samples_p: Samples from distribution P
        samples_q: Samples from distribution Q
        n_bins: Number of bins for histogram estimation

    Returns:
        Total Variation distance (between 0 and 1)
    """
    if len(samples_p) < 2 or len(samples_q) < 2:
        return 0.0

    if np.allclose(samples_p, samples_q):
        return 0.0

    var_p = np.var(samples_p)
    var_q = np.var(samples_q)
    if var_p < 1e-10 and var_q < 1e-10:
        if np.allclose(samples_p.mean(), samples_q.mean()):
            return 0.0
        else:
            return 1.0  # Maximum TV distance for different constants

    try:
        # Determine common range
        all_samples = np.concatenate([samples_p, samples_q])
        bins = np.linspace(all_samples.min(), all_samples.max(), n_bins + 1)

        # Create normalized histograms
        hist_p, _ = np.histogram(samples_p, bins=bins, density=True)
        hist_q, _ = np.histogram(samples_q, bins=bins, density=True)

        # Normalize
        sum_p = np.sum(hist_p)
        sum_q = np.sum(hist_q)
        if sum_p == 0 or sum_q == 0:
            return 0.0
        hist_p = hist_p / sum_p
        hist_q = hist_q / sum_q

        # Compute TV
        tv = 0.5 * np.sum(np.abs(hist_p - hist_q))

        return tv
    except Exception:
        return 0.0


def compute_jensen_shannon_divergence(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    n_points: int = 100
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.

    JSD is a symmetric version of KL divergence:
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    Args:
        samples_p: Samples from distribution P
        samples_q: Samples from distribution Q
        n_points: Number of points for discretization

    Returns:
        Jensen-Shannon divergence (in nats)
    """
    try:
        # Determine common support range
        all_samples = np.concatenate([samples_p, samples_q])
        x_min, x_max = all_samples.min(), all_samples.max()
        margin = 0.1 * (x_max - x_min) if x_max > x_min else 0.1
        x_min -= margin
        x_max += margin
        x_eval = np.linspace(x_min, x_max, n_points)

        # Estimate densities
        kde_p = gaussian_kde(samples_p, bw_method='scott')
        kde_q = gaussian_kde(samples_q, bw_method='scott')

        p_density = kde_p(x_eval)
        q_density = kde_q(x_eval)

        # Normalize
        p_density = p_density / np.sum(p_density)
        q_density = q_density / np.sum(q_density)

        # Compute mixture
        m_density = 0.5 * (p_density + q_density)

        # Add epsilon
        epsilon = 1e-6  # Larger epsilon for numerical stability
        p_density = np.clip(p_density, epsilon, None)
        q_density = np.clip(q_density, epsilon, None)
        m_density = np.clip(m_density, epsilon, None)

        # Re-normalize after clipping
        p_density = p_density / np.sum(p_density)
        q_density = q_density / np.sum(q_density)
        m_density = m_density / np.sum(m_density)

        # Compute JSD with bounded log ratios
        log_ratio_pm = np.clip(np.log(p_density / m_density), -20, 20)
        log_ratio_qm = np.clip(np.log(q_density / m_density), -20, 20)
        kl_pm = np.sum(p_density * log_ratio_pm)
        kl_qm = np.sum(q_density * log_ratio_qm)
        jsd = 0.5 * kl_pm + 0.5 * kl_qm

        return max(0.0, jsd)

    except Exception as e:
        warnings.warn(f"JSD computation failed: {e}. Returning 0.")
        return 0.0


METRIC_FUNCTIONS = {
    'kl_divergence': compute_kl_divergence_kde,
    'jensen_shannon': compute_jensen_shannon_divergence,
    'total_variation': compute_total_variation,
    'wasserstein': compute_wasserstein_distance,
}


def compute_consequence_metric(
    action: tuple,
    return_distributions: dict,
    metric: str = 'jensen_shannon',
    action_probs: Optional[dict] = None,
    aggregation: str = 'weighted_mean',
) -> float:
    """Compute a single consequence score for the selected metric."""
    chosen_returns = return_distributions.get(action)
    if chosen_returns is None:
        return 0.0

    metric_fn = METRIC_FUNCTIONS[metric]

    divergences = {}
    for alt_action, alt_returns in return_distributions.items():
        if alt_action != action:
            divergences[alt_action] = metric_fn(chosen_returns, alt_returns)

    if not divergences:
        return 0.0
    elif aggregation == 'max':
        return max(divergences.values())
    elif aggregation == 'mean':
        return float(np.mean(list(divergences.values())))
    elif aggregation == 'weighted_mean' and action_probs is not None:
        total_weight = sum(action_probs.get(a, 0.0) for a in divergences.keys())
        if total_weight > 0:
            return sum(
                action_probs.get(a, 0.0) * div
                for a, div in divergences.items()
            ) / total_weight
        else:
            return float(np.mean(list(divergences.values())))
    else:
        return max(divergences.values())


def compute_all_consequence_metrics(
    action: tuple,
    return_distributions: dict,
    action_probs: Optional[dict] = None,
    aggregation: str = 'weighted_mean'
) -> dict:
    """
    Compute consequence scores using all available metrics.

    Args:
        action: The chosen action
        return_distributions: Dict mapping actions to return samples (np.ndarray)
        action_probs: Optional dict mapping actions to their probabilities.
                     Used for 'weighted_mean' aggregation.
        aggregation: How to aggregate divergences across alternatives.
                    'max' - Maximum divergence (default, finds most different alternative)
                    'mean' - Simple mean of all divergences
                    'weighted_mean' - Mean weighted by action probabilities

    Returns:
        Dict mapping metric name to (consequence_score, divergences_dict)
        {
            'kl_divergence': (score, {alt_action: divergence, ...}),
            'jensen_shannon': (score, {alt_action: divergence, ...}),
            'total_variation': (score, {alt_action: distance, ...}),
            'wasserstein': (score, {alt_action: distance, ...})
        }
    """
    chosen_returns = return_distributions.get(action)

    if chosen_returns is None:
        return {
            'kl_divergence': (0.0, {}),
            'jensen_shannon': (0.0, {}),
            'total_variation': (0.0, {}),
            'wasserstein': (0.0, {})
        }

    metrics = {
        'kl_divergence': compute_kl_divergence_kde,
        'jensen_shannon': compute_jensen_shannon_divergence,
        'total_variation': compute_total_variation,
        'wasserstein': compute_wasserstein_distance
    }

    results = {}
    for metric_name, metric_fn in metrics.items():
        divergences = {}
        for alt_action, alt_returns in return_distributions.items():
            if alt_action != action:
                divergences[alt_action] = metric_fn(chosen_returns, alt_returns)

        # Aggregate divergences into a single score
        if not divergences:
            score = 0.0
        elif aggregation == 'max':
            score = max(divergences.values())
        elif aggregation == 'mean':
            score = float(np.mean(list(divergences.values())))
        elif aggregation == 'weighted_mean' and action_probs is not None:
            # Weight by probability of each alternative action
            # Higher probability alternatives contribute more to the score
            total_weight = sum(action_probs.get(a, 0.0) for a in divergences.keys())
            if total_weight > 0:
                score = sum(
                    action_probs.get(a, 0.0) * div
                    for a, div in divergences.items()
                ) / total_weight
            else:
                # Fallback to simple mean if no probs available for alternatives
                score = float(np.mean(list(divergences.values())))
        else:
            # Default fallback to max
            score = max(divergences.values())

        results[metric_name] = (score, divergences)

    return results
