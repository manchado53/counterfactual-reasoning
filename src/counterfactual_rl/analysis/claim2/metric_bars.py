"""
Why "solved" means something different in FrozenLake and in routing.

One grid, real seeds. FrozenLake has TWO possible outcomes, so "did it win?" needs no
judgement from us. Routing has a ladder of outcomes, so the win line is a choice we make --
and the two plausible choices give 5% and 92% on identical data.

Run:
    python -m counterfactual_rl.analysis.claim2.metric_bars \
        --repo <repo root> --runs-dir <runs> --out docs/figures/real/claim2
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def finals(runs_dir, glob):
    out = []
    for d in sorted(Path(runs_dir).glob(glob)):
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
    ap.add_argument('--repo', required=True)
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    z = np.load(f"{args.repo}/paper/repro/cache/claim2_frozen_lake_no_slip.npz",
                allow_pickle=True)
    fl = z['raw_1'][:, 0, -1]                 # PER, final eval per seed
    rt = finals(args.runs_dir, 'oa_per_s*')   # PER, final opt_ratio per seed

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)

    for ax, vals, title, sub in (
        (axes[0], fl, 'FrozenLake 8x8', 'reach the goal, or do not\n25 seeds, DQN+PER'),
        (axes[1], rt, 'Budget routing', 'serve 0 to 9 customers\n40 seeds, DQN+PER'),
    ):
        counts = Counter(np.round(vals, 2))
        for level, k in sorted(counts.items()):
            for i in range(k):
                col = '#2C6E49' if level >= 0.999 else ('#9E2F27' if level <= 1e-9 else '#B4600F')
                ax.plot(i % 10, level + (i // 10) * 0.012, 's', ms=9, color=col, alpha=.9)
            ax.text(-1.4, level, f'{level:.2f}', ha='right', va='center',
                    fontsize=9.5, family='monospace')
            ax.text(10.4, level, f'{k} seeds', va='center', fontsize=9, color='#48586B')
        ax.set_title(f'{title}\n{sub}', fontsize=11.5, loc='left')
        ax.set_xlim(-3.2, 13.5); ax.set_ylim(-0.09, 1.12)
        ax.set_xticks([]); ax.grid(alpha=.18, axis='y')
        for s in ('top', 'right', 'bottom'):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel('what the seed scored at the end')

    # the two candidate win lines
    for ax in axes:
        ax.axhline(0.999, color='#2C6E49', ls='--', lw=1.6)
        ax.axhline(0.005, color='#1B5FA8', ls='--', lw=1.6)
    axes[1].text(13.6, 0.999, 'OUR line:\nperfect only', fontsize=9, color='#2C6E49', va='center')
    axes[1].text(13.6, 0.02, "FrozenLake's line:\nfinished at all", fontsize=9,
                 color='#1B5FA8', va='center')

    fl_ours = 100 * (fl >= 0.999).mean(); fl_fl = 100 * (fl > 1e-9).mean()
    rt_ours = 100 * (rt >= 0.999).mean(); rt_fl = 100 * (rt > 1e-9).mean()
    axes[0].text(0.02, -0.055, f'perfect only: {fl_ours:.0f}%      finished at all: {fl_fl:.0f}%\n'
                               'the two lines AGREE — there is nothing between',
                 transform=axes[0].transAxes, fontsize=10, color='#122236')
    axes[1].text(0.02, -0.055, f'perfect only: {rt_ours:.0f}%      finished at all: {rt_fl:.0f}%\n'
                               'the two lines DISAGREE — the middle is where routing lives',
                 transform=axes[1].transAxes, fontsize=10, color='#9E2F27')

    fig.suptitle('"Solved" is not one question. FrozenLake answers it for you; routing makes '
                 'you choose.', fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_metric_bars.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")
    print(f"  FrozenLake  perfect-only {fl_ours:.0f}%   finished-at-all {fl_fl:.0f}%")
    print(f"  routing     perfect-only {rt_ours:.0f}%   finished-at-all {rt_fl:.0f}%")


if __name__ == '__main__':
    main()
