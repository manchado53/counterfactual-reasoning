"""
FINAL vs EVER vs LAST-10, on one real seed and then across all 40.

The point: training wobbles, so "did it solve the task?" depends entirely on WHICH
evaluation you look at. Two defensible choices give opposite verdicts on the same run.

  (a) one seed's 110 evaluations, with the three rules marked on it
  (b) what each rule says about all 40 PER seeds
  (c) why EVER is biased: it can only ever go up as you train longer

Run:
    python -m counterfactual_rl.analysis.claim2.metric_choice_explainer \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs --out docs/figures/real/claim2
"""

import argparse
from pathlib import Path

import numpy as np

GREEN, RED, BLUE, AMBER, INK = '#2C6E49', '#9E2F27', '#1B5FA8', '#B4600F', '#122236'


def curve(d):
    e, v = [], []
    for line in (d / 'metrics.log').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        if len(f) >= 4:
            try:
                e.append(int(f[0])); v.append(float(f[3]))
            except ValueError:
                pass
    return np.array(e), np.array(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    runs = sorted(Path(args.runs_dir).glob('oa_per_s*'))
    curves = [curve(d) for d in runs if (d / 'metrics.log').exists()]
    curves = [(e, v) for e, v in curves if v.size > 50]
    e0, v0 = curve(Path(args.runs_dir) / 'oa_per_s00')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.1], wspace=0.28)

    # (a) one seed, three rules
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(e0, v0, color='#7C8A9B', lw=1.1)
    ax.scatter(e0, v0, s=13, color='#7C8A9B', zorder=2)
    hit = np.flatnonzero(v0 >= 0.9999)
    ax.scatter(e0[hit], v0[hit], s=110, marker='*', color=GREEN, zorder=4)
    ax.annotate('EVER: it touched 1.0 here, once\n-> counts as SOLVED',
                xy=(e0[hit[0]], 1.0), xytext=(1400, 0.30), fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))
    ax.scatter([e0[-1]], [v0[-1]], s=130, color=RED, zorder=5)
    ax.annotate(f'FINAL: last check reads {v0[-1]:.2f}\n-> counts as NOT solved',
                xy=(e0[-1], v0[-1]), xytext=(3050, 0.16), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    last10 = v0[-10:].mean()
    ax.plot([e0[-10], e0[-1]], [last10, last10], color=BLUE, lw=3.5, solid_capstyle='round')
    ax.annotate(f'LAST-10 average = {last10:.2f}\n-> what it actually does',
                xy=(e0[-5], last10), xytext=(2600, 0.62), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    ax.set_xlabel('episode'); ax.set_ylabel('score at that check')
    ax.set_title('(a) ONE seed, 110 checks — three ways to grade it', fontsize=11, loc='left')
    ax.set_ylim(-0.05, 1.12); ax.grid(alpha=.25)

    # (b) the three rules across all 40 seeds
    ax = fig.add_subplot(gs[0, 1])
    fin = np.array([v[-1] for _, v in curves])
    ever = np.array([v.max() for _, v in curves])
    l10 = np.array([v[-10:].mean() for _, v in curves])
    vals = [100 * (fin >= 0.9999).mean(), 100 * (ever >= 0.9999).mean(),
            100 * (l10 >= 0.9999).mean()]
    bars = ax.bar(range(3), vals, color=[RED, GREEN, BLUE], width=0.6)
    for i, val in enumerate(vals):
        ax.text(i, val + 2, f'{val:.0f}%', ha='center', fontsize=13, weight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['FINAL\ncheck only', 'EVER\nhit 1.0', 'LAST-10\naverage'], fontsize=9.5)
    ax.set_ylabel('% of 40 PER seeds counted as "solved"')
    ax.set_title('(b) Same 40 runs, three verdicts', fontsize=11, loc='left')
    ax.set_ylim(0, 100); ax.grid(alpha=.25, axis='y')

    # (c) EVER only ever goes up
    ax = fig.add_subplot(gs[0, 2])
    grid = np.arange(500, 5501, 250)
    ever_c, fin_c = [], []
    for g in grid:
        ev = fi = 0
        for e, v in curves:
            i = np.searchsorted(e, g)
            i = min(i, len(v) - 1)
            if v[:i + 1].max() >= 0.9999: ev += 1
            if v[i] >= 0.9999: fi += 1
        ever_c.append(100 * ev / len(curves)); fin_c.append(100 * fi / len(curves))
    ax.plot(grid, ever_c, color=GREEN, lw=2.2, label='EVER (never goes down)')
    ax.plot(grid, fin_c, color=RED, lw=2.2, label='score at that moment')
    ax.fill_between(grid, fin_c, ever_c, color=AMBER, alpha=.18)
    ax.text(2100, 28, 'the gap is\npure wobble', fontsize=9, color=AMBER)
    ax.set_xlabel('where you stop training')
    ax.set_ylabel('% of seeds "solved"')
    ax.set_title('(c) EVER can only climb — that is the bias', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=.25); ax.set_ylim(-3, 100)

    fig.suptitle('"Did it solve it?" depends on which check you look at — and I calibrated on '
                 'one rule, then measured with another', fontsize=12.5, y=1.02)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_metric_choice.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")
    print(f"  FINAL {vals[0]:.0f}%   EVER {vals[1]:.0f}%   LAST-10 {vals[2]:.0f}%")


if __name__ == '__main__':
    main()
