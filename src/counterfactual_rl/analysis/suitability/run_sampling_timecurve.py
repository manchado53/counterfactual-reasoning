"""Windowed time-curve of supply-normalized replay sampling, reusing the snapshots already in
each run's sampling.npz (NO retraining). Shows whether CCE out-drills PER EARLY in training —
the window where the FL-det win is forged, which the cumulative metric buries under the long
post-convergence tail.

Per condition, averages seeds over normalized training progress (so early-stop and full runs
align). Saves docs/figures/suitability/timecurve_{det,stoch}.png.

Usage: python -m counterfactual_rl.analysis.suitability.run_sampling_timecurve
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from counterfactual_rl.analysis.claim1.frozen_lake.oracle import compute_oracle
from counterfactual_rl.analysis.suitability.rollout_sweep import compute_sampling_timecurve
from counterfactual_rl.analysis.suitability.run_realized_sampling import _run_config, _REPO

RUNS = '/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/agents/frozen_lake/runs'
GRID = np.linspace(0.0, 1.0, 20)          # common normalized-progress axis
COLORS = {'uniform': '#9e9e9e', 'PER': '#ff9800', 'CCE': '#2196F3', 'CCE-stale': '#e53935'}

DET = {'uniform': [262064, 262065, 262066], 'PER': [262067, 262068, 262069],
       'CCE': [262070, 262071, 262072], 'CCE-stale': [262073, 262074, 262075]}
STOCH = {'uniform': [262076, 262077, 262078], 'PER': [262079, 262080, 262081],
         'CCE': [262082, 262083, 262084]}


def _stakes_for(run_dir):
    cfg = _run_config(run_dir)
    _, oracle, nt = compute_oracle(map_name=cfg.get('map_name', '8x8'),
                                   is_slippery=bool(cfg.get('is_slippery', False)),
                                   gamma=float(cfg.get('gamma', 0.99)))
    nt = np.array(nt, dtype=np.int64)
    return nt, np.array([oracle[s] for s in nt], dtype=np.float64)


def _condition_curves(jobs, key):
    """Interpolate each seed's `key` series onto GRID; return (mean, lo, hi) or None."""
    rows = []
    for j in jobs:
        d = f'{RUNS}/{j}'
        nt, stakes = _stakes_for(d)
        tc = compute_sampling_timecurve(d, nt, stakes)
        if tc is None:
            continue
        prog = np.array(tc['progress']); y = np.array(tc[key], dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() < 2:
            continue
        rows.append(np.interp(GRID, prog[ok], y[ok]))
    if not rows:
        return None
    M = np.vstack(rows)
    return M.mean(0), M.min(0), M.max(0)


def _panel(ax, groups, key, title, ylabel, hline=None):
    for cond, jobs in groups.items():
        cur = _condition_curves(jobs, key)
        if cur is None:
            continue
        mean, lo, hi = cur
        ax.plot(GRID, mean, color=COLORS.get(cond, None), label=cond, lw=2)
        ax.fill_between(GRID, lo, hi, color=COLORS.get(cond, None), alpha=0.15)
    if hline is not None:
        ax.axhline(hline, color='k', lw=.5, ls='--')
    ax.set_xlabel('training progress (0=start, 1=end)')
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)


def make(env_name, groups):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    _panel(ax1, groups, 'spearman',
           f'{env_name}: does it drill high-stakes states? (per window)',
           'Spearman(oversampling, stakes)', hline=0.0)
    ax1.set_ylim(-1, 1)
    _panel(ax2, groups, 'over_top10',
           f'{env_name}: over-drill of top-10% stakes (per window)',
           'mean oversampling on top-10% stakes', hline=1.0)
    fig.suptitle(f'Windowed replay-sampling over training — {env_name}', fontweight='bold')
    fig.tight_layout()
    out = os.path.join(_REPO, 'docs', 'figures', 'suitability', f'timecurve_{env_name}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    make('det', DET)
    make('stoch', STOCH)
