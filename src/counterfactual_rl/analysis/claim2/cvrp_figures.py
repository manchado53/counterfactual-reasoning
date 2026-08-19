"""
Claim 2 figures for the routing sweep.

Four panels that together tell the whole story:
  A/B  learning curves per arm (mean + 95% bootstrap band), traffic off / on
  C    area under the learning curve, per arm, with CI  -> the overall efficiency view
  D    episodes needed to reach 0.95 of optimal          -> the speed view

The curves are drawn on a zoomed x-axis because every arm converges within ~1500 of the
14,000 training episodes — which is itself the headline finding, so the full range is
shown as an inset-style dashed marker rather than hidden.

Run:
    python -m counterfactual_rl.analysis.claim2.cvrp_figures --runs-dir <runs>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .cvrp_sweep import read_curve, steps_to, ARMS, ARM_LABEL

COLORS = {
    'uniform': '#8C9A97', 'per': '#2C7BB6',
    'cceonly': '#E39B4B', 'cceadd': '#1F7A4D', 'ccemul': '#C0392B',
}
INK = '#15211F'
PAPER = '#FFFFFF'
GRID = '#E3E9E8'
ZOOM = 3000


def load(runs_dir: Path, tag: str):
    out = {}
    for arm in ARMS:
        runs = []
        for s in range(10):
            f = runs_dir / f'c2_{arm}_n{tag}_s{s}' / 'metrics.log'
            if f.exists():
                e, v = read_curve(f)
                if e.size and e[-1] >= 14000:
                    runs.append((e, v))
        if runs:
            out[arm] = runs
    return out


def band(vals, n_boot=4000, seed=0):
    """mean curve + 95% stratified bootstrap band over seeds."""
    rng = np.random.default_rng(seed)
    a = np.stack(vals)                       # (seeds, T)
    idx = rng.integers(0, a.shape[0], (n_boot, a.shape[0]))
    boots = a[idx].mean(axis=1)              # (n_boot, T)
    return a.mean(0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = x[rng.integers(0, x.size, (n_boot, x.size))].mean(axis=1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)

    off, on = load(args.runs_dir, '00'), load(args.runs_dir, '015')
    if not off and not on:
        print('no completed runs found'); return 1

    fig = plt.figure(figsize=(13.2, 8.4))
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.20,
                          left=.07, right=.98, top=.88, bottom=.09)
    fig.suptitle('Claim 2 on routing — does consequence-based replay learn faster?',
                 fontsize=16, fontweight='bold', color=INK, y=.965)
    fig.text(.5, .925, '5 algorithms  ·  5 seeds each  ·  10-stop CVRP  ·  '
                       'shaded = 95% bootstrap CI',
             ha='center', fontsize=10.5, color='#5C6B68', family='monospace')

    # ── A / B : learning curves ──────────────────────────────────────────
    for col, (data, title) in enumerate(((off, 'Traffic OFF (deterministic)'),
                                         (on, 'Traffic ON (noise 0.15)'))):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PAPER)
        for arm, runs in data.items():
            T = min(len(v) for _, v in runs)
            eps = runs[0][0][:T]
            m, lo, hi = band([v[:T] for _, v in runs])
            k = eps <= ZOOM
            ax.plot(eps[k], m[k], color=COLORS[arm], lw=2.0, label=ARM_LABEL[arm])
            ax.fill_between(eps[k], lo[k], hi[k], color=COLORS[arm], alpha=.16, lw=0)
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel('episodes'); ax.set_ylabel('fraction of optimal')
        ax.set_ylim(0.55, 1.02); ax.set_xlim(0, ZOOM)
        ax.axhline(1.0, color='#1F7A4D', ls='--', lw=1, alpha=.7)
        ax.text(ZOOM * .99, 1.005, 'optimal', ha='right', fontsize=8.5, color='#1F7A4D')
        ax.grid(alpha=.35, color=GRID)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if col == 0:
            ax.legend(fontsize=9, loc='lower left', framealpha=.95)
        ax.annotate(f'trained to 14,000 episodes —\nall arms converge before {ZOOM}',
                    xy=(.98, .05), xycoords='axes fraction', ha='right', va='bottom',
                    fontsize=8.5, color='#8C6D1F', style='italic')

    # ── C : AUC ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor(PAPER)
    w, xs = .36, np.arange(len(ARMS))
    for off_on, (data, lbl, hatch) in enumerate(((off, 'traffic off', ''),
                                                 (on, 'traffic on', '//'))):
        for i, arm in enumerate(ARMS):
            if arm not in data:
                continue
            vals = [v.mean() for _, v in data[arm]]
            m, lo, hi = ci(vals)
            ax.bar(i + (off_on - .5) * w, m, w, color=COLORS[arm], hatch=hatch,
                   edgecolor='white', alpha=.95 if off_on == 0 else .65,
                   yerr=[[m - lo], [hi - m]], capsize=3, ecolor=INK,
                   label=lbl if i == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=8.5,
                                          rotation=12, ha='right')
    ax.set_ylabel('area under learning curve')
    ax.set_ylim(0.93, 1.0)
    ax.set_title('Overall sample efficiency (higher = better)', fontsize=11.5, color=INK)
    ax.grid(axis='y', alpha=.35, color=GRID)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8.5, loc='lower left')

    # ── D : episodes to 0.95 ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor(PAPER)
    for off_on, (data, lbl, hatch) in enumerate(((off, 'traffic off', ''),
                                                 (on, 'traffic on', '//'))):
        for i, arm in enumerate(ARMS):
            if arm not in data:
                continue
            hits = [steps_to(e, v, 0.95) for e, v in data[arm]]
            got = [h for h in hits if h is not None]
            if not got:
                continue
            m, lo, hi = ci(got)
            ax.bar(i + (off_on - .5) * w, m, w, color=COLORS[arm], hatch=hatch,
                   edgecolor='white', alpha=.95 if off_on == 0 else .65,
                   yerr=[[m - lo], [hi - m]], capsize=3, ecolor=INK,
                   label=lbl if i == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=8.5,
                                          rotation=12, ha='right')
    ax.set_ylabel('episodes to reach 0.95 of optimal')
    ax.set_title('Speed to competence (lower = better)', fontsize=11.5, color=INK)
    ax.grid(axis='y', alpha=.35, color=GRID)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8.5, loc='upper left')

    fig.text(.07, .022,
             'Verdict: no arm reliably beats DQN+PER in either condition. Every method '
             'solves the task within ~1,500 of 14,000 episodes, so there is no headroom '
             'for replay prioritisation to exploit.',
             fontsize=9.5, color='#5C6B68', style='italic')

    out = args.out or (Path(__file__).parents[3].parent / 'docs' / 'figures' / 'real' /
                       'claim2' / 'fig_c2_cvrp_sweep.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=PAPER, bbox_inches='tight')
    plt.close(fig)
    print(f'figure saved -> {out}  ({out.stat().st_size/1024:.0f} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
