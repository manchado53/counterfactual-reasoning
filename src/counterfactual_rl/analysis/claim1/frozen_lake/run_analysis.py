"""
Claim 1 FrozenLake analysis — oracle correlation + figures C1, C2, C4.

Usage:
    python -m counterfactual_rl.analysis.claim1.frozen_lake.run_analysis
    python -m counterfactual_rl.analysis.claim1.frozen_lake.run_analysis --seeds 0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .oracle import compute_oracle
from .score_states import score_all_states
from .heatmap import plot_c2_heatmaps
from ..scatter import plot_c1_scatter


def precision_at_k(oracle_vals, cce_vals, k):
    n = len(oracle_vals)
    top_n = max(1, int(n * k))
    top_oracle = set(np.argsort(oracle_vals)[-top_n:])
    top_cce    = set(np.argsort(cce_vals)[-top_n:])
    return len(top_oracle & top_cce) / top_n


def plot_c4_precision(oracle_vals, cce_by_seed, out_path, ks=(0.05, 0.10, 0.20)):
    """Bar chart: CCE Precision@K vs random baseline, trained policy, all seeds."""
    k_labels = [f'{int(k*100)}%' for k in ks]
    cce_means, cce_stds = [], []
    for k in ks:
        vals = [precision_at_k(oracle_vals, cce_by_seed[s], k) for s in cce_by_seed]
        cce_means.append(np.mean(vals))
        cce_stds.append(np.std(vals))

    x = np.arange(len(ks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x - w/2, cce_means, w, yerr=cce_stds, capsize=4,
                  color='#4CAF50', label='CCE (trained)')
    ax.bar(x + w/2, ks, w, color='#9E9E9E', label='Random baseline')

    for i, (mean, k) in enumerate(zip(cce_means, ks)):
        lift = mean / k
        ax.text(x[i] - w/2, mean + cce_stds[i] + 0.01, f'{lift:.1f}×',
                ha='center', va='bottom', fontsize=9, color='#2E7D32')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Top {l}' for l in k_labels])
    ax.set_ylabel('Precision@K')
    ax.set_ylim(0, min(1.0, max(cce_means) + max(cce_stds) + 0.15))
    ax.set_title('Claim 1 — Precision@K: CCE vs Random Baseline\n(FrozenLake 8×8, fully trained)')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Fig C4 saved → {out_path}')

CKPT_ROOT  = Path(__file__).parent / 'checkpoints'
FIGURE_DIR = Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'frozen_lake'
STAGES     = ['untrained', 'mid', 'trained']
STAGE_LABELS = ['Untrained (ep 150)', 'Mid-training (ep 3900)', 'Fully Trained (best)']

# Hole proximity coloring for C1 scatter
_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
_N = 8
_HOLE_STATES = {r * _N + c for r, row in enumerate(_MAP)
                for c, ch in enumerate(row) if ch == 'H'}


def _hole_proximity_colors(non_terminal):
    """BFS from holes to assign proximity color per state."""
    from collections import deque
    dist = {s: float('inf') for s in range(_N * _N)}
    q = deque()
    for h in _HOLE_STATES:
        dist[h] = 0
        q.append(h)
    while q:
        s = q.popleft()
        r, c = divmod(s, _N)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < _N and 0 <= nc < _N:
                ns = nr * _N + nc
                if dist[ns] == float('inf'):
                    dist[ns] = dist[s] + 1
                    q.append(ns)
    colors = {}
    for s in non_terminal:
        d = dist[s]
        if d <= 1:
            colors[s] = '#F44336'   # red — adjacent to hole
        elif d == 2:
            colors[s] = '#FF9800'   # orange — 2 steps
        else:
            colors[s] = '#2196F3'   # blue — safe
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--metric', default='total_variation')
    parser.add_argument('--n-rollouts', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Oracle
    print('Computing oracle via value iteration...')
    _, oracle, non_terminal = compute_oracle()
    print(f'  {len(non_terminal)} non-terminal states scored')

    # 2. Score all checkpoints
    cce = {}   # cce[seed][stage] = {state: score}
    for seed in args.seeds:
        cce[seed] = {}
        for stage in STAGES:
            ckpt = CKPT_ROOT / f'seed_{seed}' / f'{stage}.pkl'
            print(f'  Scoring seed={seed} stage={stage} ...')
            cce[seed][stage] = score_all_states(
                ckpt, non_terminal,
                n_rollouts=args.n_rollouts,
                horizon=args.horizon,
                gamma=args.gamma,
                metric=args.metric,
                seed=seed,
            )

    # 3. Spearman ρ table
    oracle_vals = np.array([oracle[s] for s in non_terminal])
    print(f'\n{"Stage":<12} {"Seed":>4}   {"ρ":>6}   {"p-value":>10}')
    print('-' * 40)
    rho_by_stage = {stage: [] for stage in STAGES}
    for stage in STAGES:
        for seed in args.seeds:
            cce_vals = np.array([cce[seed][stage][s] for s in non_terminal])
            rho, pval = spearmanr(oracle_vals, cce_vals)
            pval_str = f'{pval:.4f}' if pval >= 0.0001 else '<0.0001'
            print(f'{stage:<12} {seed:>4}   {rho:>6.3f}   {pval_str:>10}')
            rho_by_stage[stage].append(rho)
    print()
    for stage in STAGES:
        rhos = rho_by_stage[stage]
        print(f'{stage}: mean ρ = {np.mean(rhos):.3f} ± {np.std(rhos):.3f}')

    # 4. Precision@K (trained stage, all seeds)
    ks = (0.05, 0.10, 0.20)
    print(f'\n{"K":>5}   {"CCE":>6}   {"Random":>6}   {"Lift":>5}')
    print('-' * 32)
    for k in ks:
        vals = [precision_at_k(oracle_vals,
                               np.array([cce[s]['trained'][st] for st in non_terminal]), k)
                for s in args.seeds]
        print(f'{int(k*100):>4}%   {np.mean(vals):>6.3f}   {k:>6.3f}   {np.mean(vals)/k:>4.1f}×')

    # 5. Figures (seed 0)
    seed0 = args.seeds[0]
    colors = _hole_proximity_colors(non_terminal)

    c1_path = FIGURE_DIR / 'fig_c1_scatter_stages.png'
    plot_c1_scatter(
        oracle=oracle,
        cce_by_stage={stage: cce[seed0][stage] for stage in STAGES},
        state_colors=colors,
        stage_labels=STAGE_LABELS,
        out_path=c1_path,
    )
    print(f'\nFig C1 saved → {c1_path}')

    c2_path = FIGURE_DIR / 'fig_c2_grid_heatmaps.png'
    plot_c2_heatmaps(
        oracle=oracle,
        cce_untrained=cce[seed0]['untrained'],
        cce_trained=cce[seed0]['trained'],
        out_path=c2_path,
    )
    print(f'Fig C2 saved → {c2_path}')

    c4_path = FIGURE_DIR / 'fig_c4_precision_at_k.png'
    plot_c4_precision(
        oracle_vals=oracle_vals,
        cce_by_seed={s: np.array([cce[s]['trained'][st] for st in non_terminal])
                     for s in args.seeds},
        out_path=c4_path,
    )



if __name__ == '__main__':
    main()
