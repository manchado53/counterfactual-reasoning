"""
Claim 1, routing — the paper figure.

CCE scores a state by how much the ACTION CHOICE changed the outcome, estimated from policy
rollouts. The exact oracle says how much it really changed, by dynamic programming over every
reachable state. Claim 1 is that the two agree, and agree better as the policy improves.

  (a) CCE score vs exact oracle stakes, at three training stages  (one seed, all scored states)
  (b) Spearman rho across 10 seeds, with 95% CI
  (c) precision@k -- of the states CCE ranks in its top k%, how many are genuinely in the
      oracle's top k%. Random chance is k% itself.
  (d) robustness: the same numbers under both aggregation rules, since the shipped code
      silently used max() where the config said weighted_mean

Run:
    python -m counterfactual_rl.analysis.claim1.cvrp.paper_figure \
        --mean-dir docs/figures/real/claim1/cvrp_final_mean \
        --max-dir  docs/figures/real/claim1/cvrp_final_weighted_mean \
        --out docs/figures/real/claim1/cvrp
"""

import argparse
import json
from pathlib import Path

import numpy as np

STAGES = ['untrained', 'mid', 'trained']
SCOL = {'untrained': '#9AA5B1', 'mid': '#B4600F', 'trained': '#1B5FA8'}
INK, GREEN, RED = '#122236', '#2C6E49', '#9E2F27'


def load(d):
    return json.load(open(Path(d) / 'cvrp_claim1_results.json'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mean-dir', required=True)
    ap.add_argument('--max-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    M, X = load(args.mean_dir), load(args.max_dir)
    agg_m, agg_x = M['aggregate'], X['aggregate']
    n_seeds = agg_m['mid']['n_seeds']

    # per-state pairs for the scatter, if the pipeline saved them
    pairs = {}
    for st in STAGES:
        p = Path(args.mean_dir) / f'pairs_{st}.npz'
        if p.exists():
            z = np.load(p)
            pairs[st] = (z['oracle'], z['cce'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.0, 1.0, 1.0], wspace=0.33)

    # (a) scatter
    ax = fig.add_subplot(gs[0, 0])
    if pairs:
        for st in STAGES:
            if st not in pairs:
                continue
            o, c = pairs[st]
            j = (np.random.default_rng(0).random(len(o)) - .5) * 0.04
            ax.scatter(o + j, c, s=7, alpha=.30, color=SCOL[st],
                       label=f"{st}  rho={agg_m[st]['rho_mean']:.2f}")
        ax.set_xlabel('exact oracle stakes'); ax.set_ylabel('CCE score')
        ax.legend(fontsize=8, loc='upper left')
    else:
        for st in STAGES:
            ax.barh(STAGES.index(st), agg_m[st]['rho_mean'], color=SCOL[st], height=.55)
        ax.set_yticks(range(3)); ax.set_yticklabels(STAGES, fontsize=9)
        ax.set_xlabel('Spearman rho'); ax.set_xlim(0, 1)
    ax.set_title('(a) CCE score vs ground truth', fontsize=11, loc='left')
    ax.grid(alpha=.25)

    # (b) rho with CI
    ax = fig.add_subplot(gs[0, 1])
    mu = [agg_m[s]['rho_mean'] for s in STAGES]
    sd = [agg_m[s]['rho_std'] for s in STAGES]
    ci = [1.96 * s / np.sqrt(n_seeds) for s in sd]
    ax.errorbar(range(3), mu, yerr=ci, fmt='o-', color=INK, lw=2, ms=9, capsize=5)
    for i, v in enumerate(mu):
        ax.text(i, v + 0.045, f'{v:.2f}', ha='center', fontsize=10.5, weight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(STAGES, fontsize=9.5)
    ax.set_ylabel(r'Spearman $\rho$ (CCE, oracle)')
    ax.set_title(f'(b) Agreement rises with training\n{n_seeds} seeds, 95% CI',
                 fontsize=11, loc='left')
    ax.set_ylim(0, 1); ax.grid(alpha=.25)

    # (c) precision@k
    ax = fig.add_subplot(gs[0, 2])
    ks = [('p05_mean', 5), ('p10_mean', 10), ('p20_mean', 20)]
    w = 0.25
    for i, st in enumerate(STAGES):
        vals = [agg_m[st][k] * 100 for k, _ in ks]
        ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=SCOL[st], label=st)
    ax.plot(np.arange(3), [k for _, k in ks], 'k--', lw=1.4, marker='_',
            ms=18, label='random chance')
    ax.set_xticks(range(3)); ax.set_xticklabels([f'top {k}%' for _, k in ks], fontsize=9.5)
    ax.set_ylabel('% genuinely in the oracle top-k')
    ax.set_title('(c) precision@k', fontsize=11, loc='left')
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis='y')

    # (d) robustness to the aggregation bug
    ax = fig.add_subplot(gs[0, 3])
    xs = np.arange(3)
    ax.bar(xs - 0.19, [agg_x[s]['rho_mean'] for s in STAGES], 0.38,
           color='#C9D2DC', label="max  (what shipped)")
    ax.bar(xs + 0.19, [agg_m[s]['rho_mean'] for s in STAGES], 0.38,
           color=GREEN, label="mean (corrected)")
    ax.set_xticks(xs); ax.set_xticklabels(STAGES, fontsize=9.5)
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_title('(d) Robust to the aggregation fix', fontsize=11, loc='left')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=.25, axis='y')

    fig.suptitle('CLAIM 1 (routing): CCE\'s scores track the exact oracle, and track it better '
                 'as the policy improves', fontsize=13, y=1.03)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_c1_paper_cvrp.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f"wrote {p}")
    for st in STAGES:
        a, b = agg_m[st], agg_x[st]
        print(f"  {st:<10} mean-agg rho {a['rho_mean']:.3f}+/-{a['rho_std']:.3f}  "
              f"p@10 {a['p10_mean']:.3f}   |  max-agg rho {b['rho_mean']:.3f}")


if __name__ == '__main__':
    main()
