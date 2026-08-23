"""
The 50,000-episode rerun, in curves.

  (a) mean learning curve per arm over the full 50k — the arms stay superimposed
  (b) solve rate as a function of training budget — when saturation actually arrives,
      and where the registered E*=5500 sat relative to it
  (c) does the instability survive long training? Same seeds, first 5k vs last 5k.

Run:
    python -m counterfactual_rl.analysis.claim2.optionA_50k_figure \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs --out docs/figures/real/claim2
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ['uniform', 'per', 'cceonly', 'cceadd', 'ccemul']
LABEL = {'uniform': 'DQN-Uniform', 'per': 'DQN+PER', 'cceonly': 'DQN+CCE-only',
         'cceadd': 'CCE+TD (add)', 'ccemul': 'CCE+TD (mul)'}
COLOR = {'uniform': '#7C8A9B', 'per': '#1B5FA8', 'cceonly': '#2C6E49',
         'cceadd': '#9E2F27', 'ccemul': '#7A4FA3'}
RUN_RE = re.compile(r'^oaL_(\w+?)_s(\d+)$')


def curve(p):
    e, v = [], []
    for line in p.read_text().splitlines():
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

    data = defaultdict(list)
    for d in sorted(Path(args.runs_dir).glob('oaL_*')):
        m = RUN_RE.match(d.name)
        if not m or not (d / 'metrics.log').exists():
            continue
        e, v = curve(d / 'metrics.log')
        if e.size > 50:
            data[m.group(1)].append((int(m.group(2)), e, v))
    if not data:
        raise SystemExit('no oaL_* runs found')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 0.85], wspace=0.42)

    # (a) the two metrics on ONE panel — they disagree, and that is the point
    ax = fig.add_subplot(gs[0, 0])
    ax2 = ax.twinx()
    runs = data['per']
    n = min(len(v) for _, _, v in runs)
    eps = runs[0][1][:n]
    M = np.vstack([v[:n] for _, _, v in runs])
    mean = M.mean(axis=0)
    rate = 100 * (M >= 0.9999).mean(axis=0)
    l1, = ax.plot(eps, mean, color='#B4600F', lw=2.0)
    l2, = ax2.plot(eps, rate, color='#1B5FA8', lw=2.0)
    i15 = np.searchsorted(eps, 15000)
    ax.plot([eps[i15], eps[i15]], [0, mean[i15]], color='#122236', ls=':', lw=1.3)
    ax.annotate(f'at 15k episodes:\nmean score {mean[i15]:.2f}\nbut only '
                f'{rate[i15]:.0f}% of seeds\nare actually optimal',
                xy=(eps[i15], mean[i15]), xytext=(17000, 0.30), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=1.1, color='#122236'))
    ax.set_xlabel('episode')
    ax.set_ylabel('mean opt_ratio  (partial credit)', color='#B4600F')
    ax2.set_ylabel('% of seeds EXACTLY optimal', color='#1B5FA8')
    ax.tick_params(axis='y', colors='#B4600F'); ax2.tick_params(axis='y', colors='#1B5FA8')
    ax.set_ylim(0, 1.05); ax2.set_ylim(0, 105)
    ax.set_title('(a) Two metrics, same runs (PER)', fontsize=11, loc='left')
    ax.grid(alpha=.25)
    ax.legend([l1, l2], ['mean score', '% seeds optimal'], fontsize=8.5, loc='lower right')

    # (b) solve rate per arm
    ax = fig.add_subplot(gs[0, 1])
    grid = np.arange(1000, 50001, 1000)
    for a in ARMS:
        rs = data.get(a, [])
        if not rs:
            continue
        rates = []
        for g in grid:
            hits = sum(1 for _, e, v in rs
                       if (i := np.searchsorted(e, g)) < len(v) and v[i] >= 0.9999)
            rates.append(100 * hits / len(rs))
        ax.plot(grid, rates, color=COLOR[a], lw=1.7, label=LABEL[a])
    ax.axvline(5500, color='#122236', ls=':', lw=1.5)
    ax.text(6200, 8, 'registered\nE*=5500', fontsize=8.5, color='#122236')
    ax.set_xlabel('training budget (episodes)')
    ax.set_ylabel('% of seeds exactly optimal')
    ax.set_title('(b) All five arms, same metric', fontsize=11, loc='left')
    ax.grid(alpha=.25); ax.set_ylim(-3, 103); ax.legend(fontsize=8, loc='upper left')

    # (c) does the oscillation survive?
    ax = fig.add_subplot(gs[0, 2])
    early, late = [], []
    for a in ARMS:
        for _, e, v in data.get(a, []):
            k = np.searchsorted(e, 5000)
            if k > 5 and len(v) - k > 5:
                early.append(np.mean(np.abs(np.diff(v[:k])) > 1e-9))
                late.append(np.mean(np.abs(np.diff(v[-k:])) > 1e-9))
    ax.bar([0, 1], [100 * np.mean(early), 100 * np.mean(late)],
           color=['#B4600F', '#1B5FA8'], width=0.55)
    for x, val in zip([0, 1], [100 * np.mean(early), 100 * np.mean(late)]):
        ax.text(x, val + 1.5, f'{val:.0f}%', ha='center', fontsize=11, weight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['first\n5,000 eps', 'last\n5,000 eps'], fontsize=9)
    ax.set_ylabel('% of eval steps where the score moves')
    ax.set_title('(c) It does settle', fontsize=11, loc='left')
    ax.grid(alpha=.25, axis='y'); ax.set_ylim(0, 100)

    fig.suptitle('Strandable routing at 50,000 episodes — a high mean score is not the same '
                 'as anyone solving it', fontsize=12.5, y=1.02)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_c2_optionA_50k.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")
    print(f"  instability: first 5k = {100*np.mean(early):.1f}%, last 5k = {100*np.mean(late):.1f}%")


if __name__ == '__main__':
    main()
