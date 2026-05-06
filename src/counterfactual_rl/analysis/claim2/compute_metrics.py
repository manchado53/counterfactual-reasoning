"""Compute all Claim 2 metrics from parsed log arrays using rliable.

All rliable calls use (n_seeds, n_envs=1) as the fundamental unit.
"""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
from rliable import library as rly
from rliable import metrics as rl_metrics


ALG_ORDER = ['DQN-Uniform', 'DQN+PER', 'DQN+CCE-only', 'CCE+TD (add)', 'CCE+TD (mul)']


def iqm_curves(raw: Dict[str, np.ndarray], reps: int = 50000) -> Dict[str, Tuple]:
    """IQM learning curves with 95% bootstrap CI.

    Args:
        raw: {alg: (n_seeds, 1, n_checkpoints)}

    Returns:
        {alg: (iqm_vals, ci_lo, ci_hi)} each shape (n_checkpoints,)
    """
    algs = list(raw.keys())
    n_checkpoints = next(iter(raw.values())).shape[2]

    iqm_vals = {a: [] for a in algs}
    ci_lo = {a: [] for a in algs}
    ci_hi = {a: [] for a in algs}

    for t in range(n_checkpoints):
        scores_t = {a: raw[a][:, :, t] for a in algs}
        point, ci = rly.get_interval_estimates(scores_t, rl_metrics.aggregate_iqm, reps=reps)
        for a in algs:
            iqm_vals[a].append(float(np.squeeze(point[a])))
            ci_lo[a].append(float(np.squeeze(ci[a][0])))
            ci_hi[a].append(float(np.squeeze(ci[a][1])))

    return {a: (np.array(iqm_vals[a]), np.array(ci_lo[a]), np.array(ci_hi[a])) for a in algs}


def final_iqm(raw: Dict[str, np.ndarray], reps: int = 50000) -> Dict[str, Tuple]:
    """IQM of last 10% of checkpoints per seed/env.

    Returns:
        {alg: (point, ci_lo, ci_hi)} scalars
    """
    final_scores = {}
    for alg, arr in raw.items():
        cutoff = int(0.9 * arr.shape[2])
        final_scores[alg] = arr[:, :, cutoff:].mean(axis=2)  # (n_seeds, 1)

    point, ci = rly.get_interval_estimates(final_scores, rl_metrics.aggregate_iqm, reps=reps)
    return {
        a: (float(np.squeeze(point[a])),
            float(np.squeeze(ci[a][0])),
            float(np.squeeze(ci[a][1])))
        for a in final_scores
    }


def steps_to_threshold(
    raw: Dict[str, np.ndarray],
    eval_steps: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, Tuple]:
    """Median steps-to-threshold ± IQR across seeds.

    Returns:
        {alg: (median, iqr, n_censored)} — n_censored = seeds that never reached threshold
    """
    results = {}
    for alg, arr in raw.items():
        steps = eval_steps[alg]
        seed_steps = []
        for seed_idx in range(arr.shape[0]):
            curve = arr[seed_idx, 0, :]  # (n_checkpoints,)
            crossed = np.where(curve >= threshold)[0]
            if len(crossed) == 0:
                seed_steps.append(np.inf)
            else:
                seed_steps.append(float(steps[crossed[0]]))
        seed_steps = np.array(seed_steps)
        finite = seed_steps[np.isfinite(seed_steps)]
        n_censored = int(np.sum(~np.isfinite(seed_steps)))
        if len(finite) == 0:
            median = np.inf
            iqr = np.inf
        else:
            median = float(np.median(finite))
            iqr = float(np.subtract(*np.percentile(finite, [75, 25])))
        results[alg] = (median, iqr, n_censored)
    return results


def prob_improvement(
    raw: Dict[str, np.ndarray],
    baseline: str = 'DQN+PER',
    reps: int = 50000,
) -> Dict[str, Tuple]:
    """P(algorithm > baseline) on final win rate, per environment.

    Returns:
        {alg: (point, ci_lo, ci_hi)} for each non-baseline algorithm
    """
    final_scores = {}
    for alg, arr in raw.items():
        cutoff = int(0.9 * arr.shape[2])
        final_scores[alg] = arr[:, :, cutoff:].mean(axis=2)  # (n_seeds, 1)

    if baseline not in final_scores:
        return {}

    base_scores = final_scores[baseline]
    results = {}
    for alg, scores in final_scores.items():
        if alg == baseline:
            continue
        p_fn = lambda x, y=base_scores: rl_metrics.probability_of_improvement(x, y)
        point, ci = rly.get_interval_estimates({alg: scores}, p_fn, reps=reps)
        results[alg] = (float(np.squeeze(point[alg])),
                        float(np.squeeze(ci[alg][0])),
                        float(np.squeeze(ci[alg][1])))
    return results


def iqm_length_curves(raw_length: Dict[str, np.ndarray], reps: int = 50000) -> Dict[str, Tuple]:
    """IQM episode-length curves with 95% bootstrap CI.

    Same computation as iqm_curves() but on avg_length arrays.
    """
    return iqm_curves(raw_length, reps=reps)


def wallclock_to_threshold(
    steps_thresh: Dict[str, Tuple],
    wallclock: Dict[str, Dict],
    eval_steps: Dict[str, np.ndarray],
) -> Dict[str, Tuple]:
    """Convert median steps-to-threshold into wall-clock hours.

    Assumes uniform step rate: wc_hours = median_steps × (total_hours / total_steps).

    Returns:
        {alg: (wc_hours, n_censored)}  — wc_hours=inf if seed never crossed threshold
    """
    results = {}
    for alg in steps_thresh:
        if alg not in wallclock or alg not in eval_steps:
            continue
        median_steps, _iqr, n_cens = steps_thresh[alg]
        total_hours = wallclock[alg]['total_hours']
        steps_arr = eval_steps[alg]
        total_steps = float(steps_arr[-1]) if len(steps_arr) > 0 else 0.0
        if total_steps == 0.0 or not np.isfinite(median_steps):
            results[alg] = (np.inf, n_cens)
        else:
            rate = total_hours / total_steps  # hours per env step
            results[alg] = (float(median_steps * rate), n_cens)
    return results


def parse_wallclock_components(run_dirs: Dict[str, list]) -> Dict[str, Dict]:
    """Per-component timing breakdown averaged over seeds.

    Skips parent-level 'update' timer (its children cover its time) and 'total'
    to avoid double-counting nested timers in fig5c stacked bars.

    Returns:
        {alg: {component_name: avg_hours_per_run}}
    """
    _SKIP = {'total', 'update'}
    results = {}
    for alg, dirs in run_dirs.items():
        comp_totals: Dict[str, float] = {}
        n_runs = 0
        for d in dirs:
            timing_path = os.path.join(d, 'timing.jsonl')
            if not os.path.isfile(timing_path):
                continue
            with open(timing_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    comp = entry.get('component', '')
                    dur = entry.get('duration_s', 0.0)
                    if comp in _SKIP:
                        continue
                    comp_totals[comp] = comp_totals.get(comp, 0.0) + dur
            n_runs += 1
        if n_runs > 0:
            results[alg] = {c: v / 3600 / n_runs for c, v in comp_totals.items()}
    return results


def parse_wallclock(run_dirs: Dict[str, list]) -> Dict[str, Dict]:
    """Parse timing.jsonl files and return wall-clock sums per algorithm.

    Uses the single 'total' entry as ground-truth wall-clock.
    scoring_hours = sum of update.scoring.* leaf timers (CCE overhead only).
    training_hours = total_hours - scoring_hours.

    Args:
        run_dirs: {alg: [list of run directory paths]}

    Returns:
        {alg: {'training_hours': float, 'scoring_hours': float, 'total_hours': float}}
    """
    results = {}
    for alg, dirs in run_dirs.items():
        scoring_s, total_s = 0.0, 0.0
        n_runs = 0
        for d in dirs:
            timing_path = os.path.join(d, 'timing.jsonl')
            if not os.path.isfile(timing_path):
                continue
            with open(timing_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    comp = entry.get('component', '')
                    dur = entry.get('duration_s', 0.0)
                    if comp == 'total':
                        total_s += dur
                    elif comp.startswith('update.scoring.'):
                        scoring_s += dur
            n_runs += 1
        if n_runs > 0:
            avg_total = total_s / n_runs / 3600
            avg_scoring = scoring_s / n_runs / 3600
            results[alg] = {
                'training_hours': avg_total - avg_scoring,
                'scoring_hours': avg_scoring,
                'total_hours': avg_total,
            }
    return results


def compute_all(
    raw: Dict[str, np.ndarray],
    eval_steps: Dict[str, np.ndarray],
    threshold: float,
    raw_length: Optional[Dict[str, np.ndarray]] = None,
    run_dirs: Optional[Dict[str, list]] = None,
    reps: int = 50000,
) -> Dict:
    """Compute all Claim 2 metrics.

    Args:
        raw:        {alg: (n_seeds, 1, n_checkpoints)}
        eval_steps: {alg: (n_checkpoints,)}
        threshold:  pre-registered win-rate threshold for steps-to-threshold
        raw_length: {alg: (n_seeds, 1, n_checkpoints)} avg_length (optional)
        run_dirs:   {alg: [list of run dirs]} for wall-clock parsing (optional)
        reps:       bootstrap resamples

    Returns dict with keys:
        iqm_curves, final_iqm, steps_thresh, prob_improve,
        wallclock, iqm_length_curves, wallclock_thresh, wallclock_components
    """
    wc = parse_wallclock(run_dirs) if run_dirs else {}
    thresh = steps_to_threshold(raw, eval_steps, threshold)

    result = {
        'iqm_curves':   iqm_curves(raw, reps=reps),
        'final_iqm':    final_iqm(raw, reps=reps),
        'steps_thresh': thresh,
        'prob_improve': prob_improvement(raw, reps=reps),
        'wallclock':    wc,
    }

    if raw_length is not None:
        result['iqm_length_curves'] = iqm_length_curves(raw_length, reps=reps)
    else:
        result['iqm_length_curves'] = {}

    result['wallclock_thresh'] = (
        wallclock_to_threshold(thresh, wc, eval_steps) if wc else {}
    )
    result['wallclock_components'] = (
        parse_wallclock_components(run_dirs) if run_dirs else {}
    )

    return result
