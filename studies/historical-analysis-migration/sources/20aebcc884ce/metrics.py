"""The 6 suitability metrics — pure numpy. The single place definitions live in code
(prose definitions: docs/SUITABILITY_METRICS.md). Each returns a dict (carries CIs /
sub-components / proxy flags). Deferred Option-B metrics (precision@k, ESS) are NOT here —
they need trainer instrumentation of the realized replay draws.
"""

import warnings
from contextlib import contextmanager

import numpy as np
from scipy.stats import spearmanr


@contextmanager
def _nan_quiet():
    """Ignore NaN-slice RuntimeWarnings + FP flags (illegal C4 columns are all-NaN by design;
    numpy raises these via warnings.warn, which np.errstate does NOT cover)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

from counterfactual_rl.analysis.diagnostics.plot_diagnostics import gini, spearman_boot


def stakes_C(returns) -> np.ndarray:
    """C(s) = spread over actions of the mean rollout return.

    nan-aware: illegal/unavailable actions (Connect Four full columns) are NaN and ignored, so the
    spread is over LEGAL actions only. On FrozenLake (no NaNs) this is identical to max−min."""
    m = returns.mean(axis=2)
    with _nan_quiet():
        return (np.nanmax(m, axis=1) - np.nanmin(m, axis=1)).astype(np.float64)


def concentration(C, k_frac: float = 0.1) -> dict:
    """Gini of per-state stakes (+ top-k mass, normalized entropy). Forecast metric.

    Drops non-finite stakes (NaN where a board had no scorable spread); identical on FrozenLake."""
    C = np.asarray(C, dtype=np.float64)
    C = C[np.isfinite(C)]
    n = C.size
    if n == 0:
        return {"gini": 0.0, "topk_mass": 0.0, "entropy": 0.0, "k_frac": k_frac}
    total = C.sum()
    g = float(gini(C))
    order = np.sort(C)[::-1]
    k = max(1, int(round(k_frac * n)))
    topk_mass = float(order[:k].sum() / (total + 1e-12))
    p = C / (total + 1e-12)
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    ent_norm = float(ent / np.log(n)) if n > 1 else 0.0
    return {"gini": g, "topk_mass": topk_mass, "entropy": ent_norm, "k_frac": k_frac}


def snr(returns, eps: float = 1e-9, cap: float = 1e3) -> dict:
    """Aggregate between-action signal / within-action (environment) noise.

    Use a RATIO OF AGGREGATES, not a median of per-state ratios: most states are dead zones
    (rollouts return 0 under a partial policy) with between≈within≈0, which would drag a median
    to 0. Aggregating cancels those zeros and keeps the decision-relevant states.
    Greedy rollout continuation → `within` is environment stochasticity only. Deterministic env →
    agg_within ≈ 0 → SNR saturates at `cap` (flagged by within_zero_frac≈1)."""
    # nan-aware (illegal C4 columns are NaN → ignored); identical to .var/.mean on FrozenLake.
    with _nan_quiet():
        between = np.nanvar(np.nanmean(returns, axis=2), axis=1)   # (S,) action effect per state
        within = np.nanmean(np.nanvar(returns, axis=2), axis=1)    # (S,) env noise per state
        agg_between = float(np.nanmean(between))
        agg_within = float(np.nanmean(within))
        value = min(agg_between / (agg_within + eps), cap)
        return {
            "value": float(value),
            "agg_between": agg_between,
            "agg_within": agg_within,
            "within_zero_frac": float(np.nanmean(within <= eps)),
            "median_ratio": float(np.nanmedian(between / (within + eps))),
        }


def distinct_td(cce_priority, abs_td) -> dict:
    """1 − |Spearman(CCE, |TD|)|. High → CCE differs from PER → can beat it. GO/NO-GO."""
    cce = np.asarray(cce_priority, dtype=np.float64)
    td = np.asarray(abs_td, dtype=np.float64)
    fin = np.isfinite(cce) & np.isfinite(td)   # scipy.spearmanr does NOT drop NaN — mask first
    if fin.sum() < 2:
        return {"value": None, "spearman_cce_td": None}
    rho = spearmanr(cce[fin], td[fin]).correlation
    if rho is None or np.isnan(rho):
        return {"value": None, "spearman_cce_td": None}
    return {"value": float(1.0 - abs(rho)), "spearman_cce_td": float(rho)}


def gain_fidelity(cce_priority, qstar_spread_at_states) -> dict:
    """Spearman(CCE, exact Q*-spread). Calibration (FrozenLake only); None if no oracle."""
    if qstar_spread_at_states is None:
        return None
    rho, (lo, hi) = spearman_boot(np.asarray(qstar_spread_at_states), np.asarray(cce_priority))
    return {"value": (None if np.isnan(rho) else float(rho)), "ci": [float(lo), float(hi)]}


def need(C, d_at_states) -> dict:
    """Spearman(C(s), discounted visit freq). High → high-stakes states are revisited."""
    C = np.asarray(C, dtype=np.float64)
    d = np.asarray(d_at_states, dtype=np.float64)
    fin = np.isfinite(C) & np.isfinite(d)      # mask before spearmanr (no silent NaN)
    rho = spearmanr(C[fin], d[fin]).correlation if fin.sum() >= 2 else None
    coverage = float(np.mean(d[fin] > 0)) if fin.any() else 0.0
    return {
        "value": (None if rho is None or np.isnan(rho) else float(rho)),
        "coverage": coverage,
        "mode": "tabular",
    }


def horizon_fit(cf_horizon, means_by_h: dict, rel_tol: float = 0.05) -> dict:
    """cf_horizon / eff_horizon_proxy, proxy = smallest H where mean_s C(s) stabilizes."""
    hs = sorted(means_by_h)
    eff = hs[-1] if hs else cf_horizon
    for i in range(1, len(hs)):
        prev = means_by_h[hs[i - 1]]
        cur = means_by_h[hs[i]]
        if abs(cur - prev) / (abs(prev) + 1e-12) < rel_tol:
            eff = hs[i - 1]
            break
    value = float(cf_horizon / eff) if eff > 0 else float("nan")
    return {"value": value, "eff_horizon_proxy": float(eff), "is_proxy": True,
            "means_by_h": {int(k): float(v) for k, v in means_by_h.items()}}
