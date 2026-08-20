"""
What changed between the old routing env and the strandable one — measured, not asserted.

Three panels:
  (a) the rule that changed, drawn on the real map: the old env RESERVED the trip home, so a
      move that could strand the vehicle was simply illegal. The new env allows it, and the
      run then scores nothing.
  (b) the exact-oracle stakes distribution, both envs plus FrozenLake as the target shape.
  (c) what seeds actually end up doing, pooled over every arm in each env.

Run:
    python -m counterfactual_rl.analysis.claim2.env_before_after \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs --out docs/figures/real/claim2
"""

import argparse
import re
from pathlib import Path

import numpy as np

from counterfactual_rl.envs.routing_budget import BudgetRoutingEnv, DEPOT
from counterfactual_rl.analysis.claim1.cvrp.budget_oracle import compute_oracle, stakes

OLD_C, NEW_C, FL_C = '#7C8A9B', '#9E2F27', '#2C6E49'


def bands(st):
    st = np.asarray(st, float)
    rel = st / st.max()
    return (100 * (rel < 0.05).mean(),
            100 * ((rel >= 0.05) & (rel <= 0.50)).mean(),
            100 * (rel > 0.50).mean())


def finals(runs_dir, pattern):
    out = []
    for d in sorted(Path(runs_dir).glob(pattern)):
        f = d / 'metrics.log'
        if not f.exists():
            continue
        vals = []
        for line in f.read_text().splitlines():
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 4:
                try:
                    vals.append(float(p[3]))
                except ValueError:
                    pass
        if vals:
            out.append(vals[-1])
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    old = BudgetRoutingEnv(budget_mult=0.95, capacity=10, dist_scale=10)
    new = BudgetRoutingEnv(budget_mult=0.95, capacity=10, dist_scale=10,
                           allow_stranding=True, reward_shape='terminal')
    ob = bands(stakes(old, compute_oracle(old)[1])[old.action_masks_np.sum(1) >= 2])
    nb = bands(stakes(new, compute_oracle(new)[1])[new.action_masks_np.sum(1) >= 2])
    flb = (50.9, 0.0, 49.1)

    old_fin = finals(args.runs_dir, 'bdm_*_b100_c10_s*')
    new_fin = finals(args.runs_dir, 'oa_*')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.26)

    # ── (a) the rule, on the real map ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    xy = old.node_xy
    for j in range(1, old.n_nodes):
        ax.scatter(*xy[j], s=120, facecolors='none', edgecolors='#bbbbbb', linewidths=1.3, zorder=2)
    ax.scatter(*xy[DEPOT], marker='s', s=170, color='#222222', zorder=4)
    ax.text(xy[DEPOT, 0] - 0.07, xy[DEPOT, 1] - 0.02, 'depot', ha='right', fontsize=9)
    # a stop and a far stop whose connecting line does NOT pass through the depot,
    # otherwise the arrow and the depot label sit on top of each other
    here, far = 2, 5
    ax.scatter(*xy[here], s=150, color='#d62728', zorder=5)
    ax.text(xy[here, 0] - 0.02, xy[here, 1] + 0.05, 'truck', fontsize=9, color='#d62728')
    ax.annotate('', xy=xy[far], xytext=xy[here],
                arrowprops=dict(arrowstyle='-|>', lw=2.4, color=NEW_C, shrinkA=9, shrinkB=9))
    ax.annotate('', xy=xy[DEPOT], xytext=xy[far],
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#bbbbbb', ls=':',
                                shrinkA=9, shrinkB=9))
    mid = (xy[here] + xy[far]) / 2
    ax.text(mid[0], mid[1] - 0.07, 'drive here?', ha='center', fontsize=10, color=NEW_C)
    ax.text(xy[far, 0] + 0.02, xy[far, 1] + 0.04, 'far stop', fontsize=9, color='#666666')
    ax.text(0.02, 0.14, 'OLD:  move is ILLEGAL — the trip home was reserved,\n'
                        '         so the truck could never strand itself',
            transform=ax.transAxes, fontsize=9.5, color=OLD_C, va='top')
    ax.text(0.02, 0.06, 'NEW:  move is ALLOWED — no fuel left to get back,\n'
                        '         run ends STRANDED and scores 0',
            transform=ax.transAxes, fontsize=9.5, color=NEW_C, va='top')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.16, 1.08); ax.axis('off')
    ax.set_title('(a) The one rule that changed', fontsize=11, loc='left')

    # ── (b) stakes distribution ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    rows = [('FrozenLake det\n(CCE wins)', flb), ('OLD routing', ob), ('NEW strandable', nb)]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, b) in zip(ys, rows):
        left = 0
        for val, col, nm in zip(b, ['#D3DAE3', '#B4600F', '#1B5FA8'],
                                ['barely', 'middle', 'critical']):
            ax.barh(y, val, left=left, color=col, height=0.55,
                    label=nm if y == ys[0] else None)
            if val >= 8:
                ax.text(left + val / 2, y, f'{val:.0f}', ha='center', va='center',
                        fontsize=9, color='white')
            left += val
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel('% of decision states'); ax.set_xlim(0, 100)
    ax.set_title('(b) How much the choice matters (exact oracle)', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.30))
    ax.grid(alpha=.2, axis='x')

    # ── (c) where seeds actually end up ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-0.02, 1.02, 26)
    ax.hist(old_fin, bins=bins, color=OLD_C, alpha=0.75,
            label=f'OLD  (n={len(old_fin)})')
    ax.hist(new_fin, bins=bins, color=NEW_C, alpha=0.6,
            label=f'NEW  (n={len(new_fin)})')
    ax.axvline(0.0, color=NEW_C, ls='--', lw=1.2)
    ax.text(0.03, ax.get_ylim()[1] * 0.92, 'total failure\nnow reachable',
            fontsize=8.5, color=NEW_C)
    ax.set_xlabel('final opt_ratio'); ax.set_ylabel('runs')
    ax.set_title('(c) Where runs end up, all arms pooled', fontsize=11, loc='left')
    ax.legend(fontsize=9); ax.grid(alpha=.25, axis='y')

    fig.suptitle('Budget routing, before and after Option A — one rule change, '
                 'measured three ways', fontsize=12.5, y=1.01)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_env_before_after.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")
    print(f"  OLD  stakes {ob[0]:.1f}/{ob[1]:.1f}/{ob[2]:.1f}  failure states {old.n_failure_states}")
    print(f"  NEW  stakes {nb[0]:.1f}/{nb[1]:.1f}/{nb[2]:.1f}  failure states {new.n_failure_states:,}")
    print(f"  OLD finals: mean {old_fin.mean():.3f}, {100*(old_fin>=0.9999).mean():.0f}% optimal, "
          f"{100*(old_fin<=1e-9).mean():.0f}% zero")
    print(f"  NEW finals: mean {new_fin.mean():.3f}, {100*(new_fin>=0.9999).mean():.0f}% optimal, "
          f"{100*(new_fin<=1e-9).mean():.0f}% zero")


if __name__ == '__main__':
    main()
