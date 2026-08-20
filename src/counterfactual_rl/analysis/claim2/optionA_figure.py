"""
Option A training figure — what 200 runs on the strandable routing env actually look like.

Three panels, because the summary numbers hide the thing that matters:
  (a) mean learning curve per arm — the arms are indistinguishable
  (b) individual seeds — the greedy policy never converges, it oscillates between
      near-optimal and total loss for the whole run
  (c) the two solve-rate definitions side by side, which is why the pre-registered
      metric read 5% where the calibration promised 50%

Run:
    python -m counterfactual_rl.analysis.claim2.optionA_figure \
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
RUN_RE = re.compile(r'^oa_(\w+?)_s(\d+)$')


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


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    data = defaultdict(list)
    for d in sorted(Path(args.runs_dir).glob('oa_*')):
        m = RUN_RE.match(d.name)
        if not m or not (d / 'metrics.log').exists():
            continue
        e, v = curve(d / 'metrics.log')
        if e.size:
            data[m.group(1)].append((int(m.group(2)), e, v))
    if not data:
        raise SystemExit('no oa_* runs found')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.26)

    # (a) mean curves — all five arms
    ax = fig.add_subplot(gs[0, 0])
    for a in ARMS:
        runs = data.get(a, [])
        if not runs:
            continue
        n = min(len(v) for _, _, v in runs)
        eps = runs[0][1][:n]
        M = np.vstack([v[:n] for _, _, v in runs])
        mean = M.mean(axis=0)
        sem = M.std(axis=0) / np.sqrt(M.shape[0])
        ax.plot(eps, mean, color=COLOR[a], lw=1.7, label=f"{LABEL[a]} (n={len(runs)})")
        ax.fill_between(eps, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=COLOR[a], alpha=0.13, lw=0)
    ax.set_xlabel('episode'); ax.set_ylabel('opt_ratio  (served / max servable)')
    ax.set_title('(a) All five arms, mean of 40 seeds', fontsize=11, loc='left')
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc='lower right'); ax.set_ylim(0, 1.05)

    # (b) individual seeds — the actual story
    ax = fig.add_subplot(gs[0, 1])
    runs = sorted(data['per'])[:4]
    for i, (seed, e, v) in enumerate(runs):
        ax.plot(e, v + i * 0.02, lw=1.0, alpha=0.85, label=f'seed {seed}')
    ax.axhline(1.0, color='#2C6E49', ls='--', lw=1.2)
    ax.text(200, 1.02, 'the optimum', fontsize=8.5, color='#2C6E49')
    ax.axhline(0.0, color='#9E2F27', ls='--', lw=1.2)
    ax.text(200, 0.03, 'stranded — run scores nothing', fontsize=8.5, color='#9E2F27')
    ax.set_xlabel('episode'); ax.set_ylabel('opt_ratio')
    ax.set_title('(b) Four individual PER seeds — it never settles', fontsize=11, loc='left')
    ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2, loc='lower right'); ax.set_ylim(-0.05, 1.12)

    # (c) the two definitions of "solved"
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(len(ARMS)); w = 0.36
    for j, (mode, hatch) in enumerate((('final', None), ('ever', '//'))):
        rates, errs = [], [[], []]
        for a in ARMS:
            runs = data.get(a, [])
            n = len(runs)
            k = sum(1 for _, _, v in runs
                    if (v[-1] if mode == 'final' else v.max()) >= 0.9999)
            r = 100 * k / n if n else 0
            lo, hi = wilson(k, n)
            rates.append(r); errs[0].append(r - lo); errs[1].append(hi - r)
        ax.bar(xs + (j - 0.5) * w, rates, w, yerr=errs, capsize=3,
               color=[COLOR[a] for a in ARMS], alpha=1.0 if j == 0 else 0.45,
               hatch=hatch, edgecolor='white',
               label='FINAL policy is optimal' if j == 0 else 'EVER reached optimal')
    SHORT = {'uniform': 'Uniform', 'per': 'PER', 'cceonly': 'CCE-only',
             'cceadd': 'CCE+TD\nadd', 'ccemul': 'CCE+TD\nmul'}
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[a] for a in ARMS], fontsize=8.5)
    ax.set_ylabel('% of 40 seeds')
    ax.set_title('(c) Two definitions of "solved"', fontsize=11, loc='left')
    ax.grid(alpha=.25, axis='y'); ax.legend(fontsize=8.5, loc='upper left'); ax.set_ylim(0, 100)

    fig.suptitle('OPTION A — strandable budget routing, 200 runs, pre-registered at E*=5500.  '
                 'Making failure possible made the policy unstable, not CCE better.',
                 fontsize=12.5, y=1.02)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_c2_optionA_training.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")


if __name__ == '__main__':
    main()
