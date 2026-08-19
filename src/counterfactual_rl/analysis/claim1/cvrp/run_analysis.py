"""
Claim 1 for routing (CVRP / TSP) — does CCE find the decisions that matter?

Compares CCE scores (computed only from policy rollouts) against the EXACT oracle
(dynamic programming on the known transition table — no learner involved) at three
training stages, and reports Spearman rho + Precision@K, mirroring the FrozenLake
pipeline so the numbers are directly comparable.

Usage:
    python -m counterfactual_rl.analysis.claim1.cvrp.run_analysis \
        --run-dir <path/to/runs/JOBID> [--instance default] [--capacity 10]

Expects checkpoints under <run-dir>/checkpoints/ (plus best.pkl). Three stages are
picked automatically: earliest ckpt (untrained), middle ckpt (mid), best.pkl (trained).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from counterfactual_rl.agents.cvrp.dqn import CVRPDQN, build_env
from counterfactual_rl.agents.cvrp.config import DEFAULT_CONFIG
from counterfactual_rl.envs.cvrp import DEPOT
from .oracle import compute_oracle
from .score_states import score_states
from ..scatter import plot_c1_scatter

STAGES = ['untrained', 'mid', 'trained']


def precision_at_k(oracle_vals, cce_vals, k):
    n = len(oracle_vals)
    top_n = max(1, int(n * k))
    top_oracle = set(np.argsort(oracle_vals)[-top_n:])
    top_cce = set(np.argsort(cce_vals)[-top_n:])
    return len(top_oracle & top_cce) / top_n


def pick_checkpoints(run_dir: Path):
    """
    (untrained, mid, trained) checkpoint paths from a run directory.

    Uses the LAST checkpoint for 'trained', not best.pkl. Once the agent reaches a
    perfect score, best.pkl can never improve on it (the update is a strict >), so it
    freezes at whatever episode first hit the ceiling — which silently made 'mid' and
    'trained' the same weights and reported one measurement twice.
    """
    ckpts = sorted((run_dir / 'checkpoints').glob('ckpt_*.pkl'))
    if len(ckpts) < 3:
        raise FileNotFoundError(
            f"need >= 3 checkpoints under {run_dir / 'checkpoints'}, found {len(ckpts)}")
    chosen = {'untrained': ckpts[0], 'mid': ckpts[len(ckpts) // 2], 'trained': ckpts[-1]}

    # Guard against silently comparing identical weights.
    import hashlib
    digests = {k: hashlib.md5(p.read_bytes()).hexdigest() for k, p in chosen.items()}
    if len(set(digests.values())) < 3:
        dupes = [k for k in chosen if list(digests.values()).count(digests[k]) > 1]
        print(f'  WARNING: identical checkpoints across stages {dupes} in {run_dir.name} '
              '— those stages are not independent measurements.')
    return chosen


def state_colors_by_position(env, states):
    """
    Colour each decision state by WHERE the choice is made — the routing analogue of
    FrozenLake's hole-proximity colouring.

      red    : at a customer with the depot in play  -> the "reload or continue?" call
      blue   : at the depot                          -> routine "which stop first?"
      orange : at a customer, depot not an option    -> pure next-stop choice
    """
    masks = np.asarray(env.action_masks)
    colors = {}
    for s in states:
        at_depot = int(env.state_current_np[s]) == DEPOT
        if at_depot:
            colors[s] = '#2196F3'
        elif masks[s, DEPOT]:
            colors[s] = '#F44336'
        else:
            colors[s] = '#FF9800'
    return colors


def plot_importance_map(env, oracle, cce, out_path, optimal_tour=None):
    """
    Map view: node layout with the optimal plan, plus per-first-leg stakes.

    Panel 1 — the instance and its optimal plan.
    Panel 2 — oracle stakes of the FIRST decision (leaving the depot), per destination.
    Panel 3 — the same first decision scored by CCE.
    """
    xy = env.node_xy
    start = env.start_states[0]
    masks = np.asarray(env.action_masks)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    if optimal_tour:
        pts = np.array([xy[i] for i in optimal_tour])
        ax.plot(pts[:, 0], pts[:, 1], '-', color='#0E7C86', lw=1.8, alpha=.9, zorder=1)
    ax.scatter(xy[1:, 0], xy[1:, 1], s=np.asarray(env.demand[1:]) * 26 + 40,
               c='#0E7C86', zorder=3, edgecolors='white', linewidths=1.2)
    ax.scatter(xy[0, 0], xy[0, 1], marker='s', s=190, c='#15211F', zorder=4)
    for i in range(1, env.n_nodes):
        ax.annotate(f'C{i}', xy[i], textcoords='offset points', xytext=(9, 7), fontsize=8)
    ax.annotate('DEPOT', xy[0], textcoords='offset points', xytext=(9, 9),
                fontsize=8, fontweight='bold')
    title = f'Instance ({env.n_customers} stops'
    title += f', capacity {env.capacity})' if env.is_capacitated else ', TSP)'
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    # Stakes of each possible FIRST move out of the depot.
    legal_first = np.flatnonzero(masks[start])
    next_s = np.asarray(env.next_states)[:, :, 0]
    for ax, source, label in ((axes[1], oracle, 'Oracle stakes'),
                              (axes[2], cce, 'CCE score')):
        vals, pts = [], []
        for a in legal_first:
            s2 = int(next_s[start, a])
            if s2 in source:
                vals.append(source[s2]); pts.append(xy[a])
        if vals:
            pts = np.array(pts); vals = np.array(vals)
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, s=170, cmap='YlOrRd',
                            edgecolors='#333', linewidths=.7, zorder=3)
            plt.colorbar(sc, ax=ax, fraction=.046)
        ax.scatter(xy[0, 0], xy[0, 1], marker='s', s=190, c='#15211F', zorder=4)
        ax.set_title(f'{label} after the first leg', fontsize=11)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'importance map saved -> {out_path}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path, nargs='+',
                    help='one or more run directories (one per seed); results are '
                         'aggregated as mean +/- std across seeds')
    ap.add_argument('--instance', default='default')
    ap.add_argument('--capacity', type=int, default=None,
                    help='load limit; -1 for TSP mode; omit for the instance default')
    ap.add_argument('--travel-noise', type=float, default=0.15,
                    help='traffic noise used for CCE ROLLOUTS. Must be > 0: under '
                         'determinism the total-variation score is degenerate (point-mass '
                         'returns -> TV collapses to 0/1 and C(s) is constant). Zero-mean, '
                         'so the exact oracle and optimal plan are unaffected.')
    ap.add_argument('--metric', default='total_variation')
    ap.add_argument('--n-rollouts', type=int, default=20)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--max-states', type=int, default=1500,
                    help='sample this many decision states (routing has tens of thousands)')
    ap.add_argument('--chunk-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG.copy()
    cfg['instance'] = args.instance
    if args.capacity is not None:
        cfg['capacity'] = None if args.capacity < 0 else args.capacity

    # The ORACLE is computed on the deterministic env (expected costs — exact ground
    # truth). CCE SCORING uses the noisy env so rollout returns actually vary. Transitions
    # are identical either way, so the same checkpoint is valid in both.
    oracle_cfg = dict(cfg, travel_noise=0.0)
    cfg['travel_noise'] = args.travel_noise
    if args.travel_noise <= 0:
        print('WARNING: travel_noise=0 makes the total-variation CCE score degenerate '
              '(constant C(s), undefined correlation). Use > 0 for Claim 1.')

    env = build_env(oracle_cfg)
    out_dir = args.out_dir or (Path(__file__).parents[5] / 'docs' / 'figures' /
                               'real' / 'claim1' / 'cvrp')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Env: {env.n_customers} customers, capacity={env.capacity}, '
          f'{env.n_states} states')

    # ── Oracle (exact, no learner involved) ──────────────────────────────────
    print('Computing exact oracle (dynamic programming)...')
    _, oracle_all, decision_states = compute_oracle(env, gamma=args.gamma)
    print(f'  {len(decision_states)} decision states')

    # Sample if the instance is large — scoring every state is unnecessary and slow.
    rng = np.random.default_rng(args.seed)
    if args.max_states and len(decision_states) > args.max_states:
        sampled = sorted(rng.choice(decision_states, args.max_states, replace=False).tolist())
        print(f'  sampling {len(sampled)} of them for scoring')
    else:
        sampled = list(decision_states)
    oracle = {s: oracle_all[s] for s in sampled}

    # ── CCE scores at three training stages, for every seed ──────────────────
    per_seed, cce_by_stage = [], {}
    for si, run_dir in enumerate(args.run_dir):
        print(f'\n=== seed {si}: {run_dir.name} ===')
        ckpts = pick_checkpoints(run_dir)
        seed_res = {}
        for stage in STAGES:
            print(f'Scoring CCE [{stage}] from {ckpts[stage].name} ...')
            agent = CVRPDQN(cfg)
            agent.load(str(ckpts[stage]))
            cce = score_states(
                None, sampled, config=cfg, agent=agent,
                n_rollouts=args.n_rollouts, horizon=args.horizon, gamma=args.gamma,
                metric=args.metric, seed=args.seed + si, chunk_size=args.chunk_size,
            )
            if si == 0:
                cce_by_stage[stage] = cce   # first seed drives the scatter figure

            common = [s for s in sampled if s in cce]
            o = np.array([oracle[s] for s in common])
            c = np.array([cce[s] for s in common])
            rho, p = spearmanr(o, c)
            prec = {f'p@{int(k*100)}': precision_at_k(o, c, k) for k in (0.05, 0.10, 0.20)}
            seed_res[stage] = dict(n=len(common), rho=float(rho), p=float(p), **prec)
            print(f'  rho={rho:.3f} (p={p:.2e})  ' +
                  '  '.join(f'{k}={v:.3f}' for k, v in prec.items()))
        per_seed.append(dict(run=run_dir.name, stages=seed_res))

    # aggregate across seeds
    results = {}
    for stage in STAGES:
        rhos = np.array([s['stages'][stage]['rho'] for s in per_seed])
        p10s = np.array([s['stages'][stage]['p@10'] for s in per_seed])
        p05s = np.array([s['stages'][stage]['p@5'] for s in per_seed])
        p20s = np.array([s['stages'][stage]['p@20'] for s in per_seed])
        results[stage] = dict(
            n_seeds=len(per_seed),
            rho_mean=float(rhos.mean()), rho_std=float(rhos.std()),
            p05_mean=float(p05s.mean()), p10_mean=float(p10s.mean()),
            p10_std=float(p10s.std()), p20_mean=float(p20s.mean()),
            max_p=float(max(s['stages'][stage]['p'] for s in per_seed)),
        )

    # ── Figures ──────────────────────────────────────────────────────────────
    colors = state_colors_by_position(env, sampled)
    labels = ['Untrained', 'Mid-training', 'Fully trained']
    scatter_path = out_dir / 'fig_c1_scatter_cvrp.png'
    plot_c1_scatter(oracle, cce_by_stage, colors, labels, scatter_path)
    print(f'scatter saved -> {scatter_path}')

    from .oracle import optimal_tour
    tour, _ = optimal_tour(env, gamma=1.0)
    plot_importance_map(env, oracle, cce_by_stage['trained'],
                        out_dir / 'fig_c1_map_cvrp.png', optimal_tour=tour)

    summary = {
        'instance': args.instance,
        'capacity': env.capacity,
        'n_customers': env.n_customers,
        'n_states': env.n_states,
        'n_decision_states': len(decision_states),
        'n_scored': len(sampled),
        'metric': args.metric,
        'n_rollouts': args.n_rollouts,
        'horizon': args.horizon,
        'travel_noise': args.travel_noise,
        'n_seeds': len(per_seed),
        'aggregate': results,
        'per_seed': per_seed,
    }
    out_json = out_dir / 'cvrp_claim1_results.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f'results saved -> {out_json}')

    print(f'\nSUMMARY over {len(per_seed)} seed(s) — Spearman rho should RISE with training:')
    for stage in STAGES:
        r = results[stage]
        print(f'  {stage:<11} rho={r["rho_mean"]:+.3f} +/- {r["rho_std"]:.3f}   '
              f'p@10={r["p10_mean"]:.3f} +/- {r["p10_std"]:.3f}   '
              f'(worst p={r["max_p"]:.1e})')
    print(f'  random-chance baseline for p@10 = 0.100')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
