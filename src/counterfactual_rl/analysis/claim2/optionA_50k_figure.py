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

    fig = plt.figure(figsize=(16, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 1.0], wspace=0.27)

    # (a) mean curves
    ax = fig.add_subplot(gs[0, 0])
    for a in ARMS:
        runs = data.get(a, [])
        if not runs:
            continue
        n = min(len(v) for _, _, v in runs)
        eps = runs[0][1][:n]
        M = np.vstack([v[:n] for _, _, v in runs])
        mean = M.mean(axis=0); sem = M.std(axis=0) / np.sqrt(M.shape[0])
        ax.plot(eps, mean, color=COLOR[a], lw=1.6, label=f'{LABEL[a]} (n={len(runs)})')
        ax.fill_between(eps, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=COLOR[a], alpha=0.12, lw=0)
    ax.axvline(5500, color='#122236', ls=':', lw=1.5)
    ax.text(6500, 0.35, 'registered\nE*=5500', fontsize=8.5, color='#122236')
    ax.set_xlabel('episode'); ax.set_ylabel('opt_ratio')
    ax.set_title('(a) 50,000 episodes, mean of 40 seeds', fontsize=11, loc='left')
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc='lower right'); ax.set_ylim(0, 1.05)

    # (b) solve rate vs training budget
    ax = fig.add_subplot(gs[0, 1])
    grid = np.arange(1000, 50001, 1000)
    for a in ARMS:
        runs = data.get(a, [])
        if not runs:
            continue
        rates = []
        for g in grid:
            hits = sum(1 for _, e, v in runs
                       if (i := np.searchsorted(e, g)) < len(v) and v[i] >= 0.9999)
            rates.append(100 * hits / len(runs))
        ax.plot(grid, rates, color=COLOR[a], lw=1.7, label=LABEL[a])
    ax.axvspan(0, 8000, color='#2C6E49', alpha=0.10)
    ax.text(600, 12, 'where arms\ncould differ', fontsize=8.5, color='#2C6E49')
    ax.axvline(5500, color='#122236', ls=':', lw=1.5)
    ax.set_xlabel('training budget (episodes)')
    ax.set_ylabel('% of seeds whose policy is optimal')
    ax.set_title('(b) Saturation arrives by ~15k and never leaves', fontsize=11, loc='left')
    ax.grid(alpha=.25); ax.set_ylim(-3, 103)

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
    ax.set_xticklabels(['first 5,000\nepisodes', 'last 5,000\nepisodes'], fontsize=9.5)
    ax.set_ylabel('% of eval steps where the score CHANGES')
    ax.set_title('(c) The policy does settle, eventually', fontsize=11, loc='left')
    ax.grid(alpha=.25, axis='y'); ax.set_ylim(0, 100)

    fig.suptitle('Strandable routing at 50,000 episodes — every arm reaches the ceiling; '
                 'random replay is the best of them', fontsize=12.5, y=1.02)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_c2_optionA_50k.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")
    print(f"  instability: first 5k = {100*np.mean(early):.1f}%, last 5k = {100*np.mean(late):.1f}%")


if __name__ == '__main__':
    main()
