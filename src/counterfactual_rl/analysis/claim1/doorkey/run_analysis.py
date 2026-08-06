"""
Claim 1 for DoorKey: do CCE scores (from policy rollouts) correlate with the exact
value-iteration oracle?

Mirrors analysis/claim1/frozen_lake/run_analysis.py. For each seed we pick three
training-stage checkpoints (untrained / mid / trained) from a training run directory,
CCE-score every reachable non-terminal state, and report Spearman rho + Precision@K
against the oracle. Figures: the shared 3-panel scatter and a DoorKey heatmap.

Run (after training 3 seeds on slip=0.2):
    python -m counterfactual_rl.analysis.claim1.doorkey.run_analysis \
        --run-dirs <run0> <run1> <run2> --seeds 0 1 2 --slip 0.2
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from counterfactual_rl.analysis.claim1.doorkey.oracle import compute_oracle
from counterfactual_rl.analysis.claim1.doorkey.score_states import score_all_states
from counterfactual_rl.analysis.claim1.doorkey.heatmap import plot_c2_heatmaps
from counterfactual_rl.analysis.claim1.scatter import plot_c1_scatter
from counterfactual_rl.envs.doorkey import DOOR_OPEN

STAGES = ['untrained', 'mid', 'trained']
STAGE_LABELS = ['Untrained', 'Mid-training', 'Fully Trained']
FIGURE_DIR = Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'doorkey'


def precision_at_k(oracle_vals, cce_vals, k):
    n = len(oracle_vals)
    top_n = max(1, int(n * k))
    top_oracle = set(np.argsort(oracle_vals)[-top_n:])
    top_cce = set(np.argsort(cce_vals)[-top_n:])
    return len(top_oracle & top_cce) / top_n


def _ckpt_episode(path):
    m = re.search(r'ckpt_(\d+)\.pkl', os.path.basename(path))
    return int(m.group(1)) if m else -1


def pick_stage_ckpts(run_dir):
    """untrained = earliest ckpt; trained = best.pkl (fallback latest); mid ~ 25% episode."""
    ckpts = sorted(glob.glob(os.path.join(run_dir, 'checkpoints', 'ckpt_*.pkl')),
                   key=_ckpt_episode)
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints under {run_dir}/checkpoints/')
    best = os.path.join(run_dir, 'best.pkl')
    trained = best if os.path.exists(best) else ckpts[-1]
    max_ep = _ckpt_episode(ckpts[-1])
    target = 0.25 * max_ep
    mid = min(ckpts, key=lambda p: abs(_ckpt_episode(p) - target))
    return {'untrained': ckpts[0], 'mid': mid, 'trained': trained}


def _phase_colors(non_terminal, env):
    """Color states by DoorKey phase: needs-key / has-key-door-shut / door-open."""
    colors = {}
    for s in non_terminal:
        _, _, _, has_key, door = env._order[s]
        if has_key == 0:
            colors[s] = '#d62728'          # red — must still get the key
        elif door != DOOR_OPEN:
            colors[s] = '#ff7f0e'          # orange — has key, door not yet open
        else:
            colors[s] = '#1f77b4'          # blue — door open, en route to goal
    return colors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dirs', nargs='+', required=True,
                   help='One training run directory per seed (must contain checkpoints/).')
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='Seed labels for the run dirs (default 0..n-1).')
    p.add_argument('--slip', type=float, default=0.2,
                   help='Slip prob for the oracle (match the training slip).')
    p.add_argument('--metric', default='total_variation')
    p.add_argument('--n-rollouts', type=int, default=100)
    p.add_argument('--horizon', type=int, default=60)
    p.add_argument('--gamma', type=float, default=0.99)
    args = p.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(len(args.run_dirs)))
    assert len(seeds) == len(args.run_dirs), 'seeds and run-dirs length mismatch'
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Oracle (on the slippery DoorKey MDP the agent is scored on)
    print(f'Computing DoorKey oracle via value iteration (slip={args.slip}) ...')
    _, oracle, non_terminal, env = compute_oracle('6x6', slip_prob=args.slip, gamma=args.gamma)
    print(f'  {len(non_terminal)} reachable non-terminal states')

    # 2. Score all checkpoints
    cce = {}  # cce[seed][stage] = {state: score}
    for seed, run_dir in zip(seeds, args.run_dirs):
        stage_ckpts = pick_stage_ckpts(run_dir)
        cce[seed] = {}
        for stage in STAGES:
            ck = stage_ckpts[stage]
            print(f'  Scoring seed={seed} stage={stage:9s} ({os.path.basename(ck)}) ...')
            cce[seed][stage] = score_all_states(
                ck, non_terminal,
                n_rollouts=args.n_rollouts, horizon=args.horizon,
                gamma=args.gamma, metric=args.metric, seed=seed,
                slip_prob=args.slip,
            )

    # 3. Spearman rho table
    oracle_vals = np.array([oracle[s] for s in non_terminal])
    print(f'\n{"Stage":<14} {"Seed":>4}   {"rho":>6}   {"p-value":>10}')
    print('-' * 42)
    rho_by_stage = {stage: [] for stage in STAGES}
    for stage in STAGES:
        for seed in seeds:
            cce_vals = np.array([cce[seed][stage][s] for s in non_terminal])
            rho, pval = spearmanr(oracle_vals, cce_vals)
            pstr = f'{pval:.4f}' if pval >= 1e-4 else '<0.0001'
            print(f'{stage:<14} {seed:>4}   {rho:>6.3f}   {pstr:>10}')
            rho_by_stage[stage].append(rho)
    print()
    for stage in STAGES:
        rhos = rho_by_stage[stage]
        print(f'{stage}: mean rho = {np.mean(rhos):.3f} +/- {np.std(rhos):.3f}')

    # 4. Precision@K (trained stage)
    print(f'\n{"K":>5}   {"CCE":>6}   {"Random":>6}   {"Lift":>5}')
    print('-' * 32)
    for k in (0.05, 0.10, 0.20):
        vals = [precision_at_k(oracle_vals,
                               np.array([cce[s]['trained'][st] for st in non_terminal]), k)
                for s in seeds]
        print(f'{int(k*100):>4}%   {np.mean(vals):>6.3f}   {k:>6.3f}   {np.mean(vals)/k:>4.1f}x')

    # 5. Figures (first seed)
    s0 = seeds[0]
    colors = _phase_colors(non_terminal, env)
    scatter_path = FIGURE_DIR / 'fig_c1_scatter_stages.png'
    plot_c1_scatter(
        oracle,
        {stage: cce[s0][stage] for stage in STAGES},
        colors, STAGE_LABELS, scatter_path,
    )
    print(f'\nSaved {scatter_path}')

    heatmap_path = FIGURE_DIR / 'fig_c2_grid_heatmaps.png'
    plot_c2_heatmaps(oracle, cce[s0]['untrained'], cce[s0]['trained'], env, heatmap_path)
    print(f'Saved {heatmap_path}')


if __name__ == '__main__':
    main()
