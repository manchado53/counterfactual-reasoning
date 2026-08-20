"""
Solve-rate analysis — the FrozenLake-style metric, recovered from existing runs.

WHY THIS EXISTS
---------------
FrozenLake's headline Claim-2 result is a per-seed PASS/FAIL: CCE-mul solved 80% of seeds
against PER's 48%. That works as a comparison because the baseline sits near the middle of
the range, so a replay strategy has room to move it.

The routing sweeps used AUC of opt_ratio over 4000 episodes instead, and by episode ~1600
every arm solves every seed. The metric is pinned at 100%, which is why 2000 runs produced
differences of fractions of a percent. The task is not hard — the MEASUREMENT was taken
long after the discriminating window closed.

This module recomputes the FrozenLake metric from the curves already on disk: for each seed,
the episode at which the greedy policy first reaches the exact oracle optimum. From that one
number per run you get the solve rate at any training budget, for free.

Use it to (a) show that the evaluation point was saturated, and (b) CHOOSE an honest
evaluation budget for the next sweep — the episode where the REFERENCE arm solves ~50%,
picked from the reference curve alone, before looking at any CCE arm.

Run:
    python -m counterfactual_rl.analysis.claim2.cvrp_solve_rate \
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
RUN_RE = re.compile(r'^bdm_(\w+?)_b(\d+)_c(\d+)_s(\d+)$')
NEVER = 10 ** 9


def read_curve(path):
    e, v = [], []
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        if len(f) >= 4:
            try:
                e.append(int(f[0])); v.append(float(f[3]))
            except ValueError:
                pass
    return np.array(e), np.array(v)


def collect_solve_episodes(runs_dir: Path):
    """{(budget, capacity): {arm: [episode-first-optimal per seed]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for d in sorted(runs_dir.glob('bdm_*')):
        m = RUN_RE.match(d.name)
        f = d / 'metrics.log'
        if not m or not f.exists():
            continue
        e, v = read_curve(f)
        if e.size < 5:
            continue
        hit = np.flatnonzero(v >= 0.9999)
        out[(int(m.group(2)) / 100, int(m.group(3)))][m.group(1)].append(
            int(e[hit[0]]) if hit.size else NEVER)
    return out


def rate(eps, budget):
    a = np.asarray(eps)
    return 100.0 * (a <= budget).mean() if a.size else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference', default='per',
                    help='arm whose curve sets the evaluation budget (default: per)')
    ap.add_argument('--target-rate', type=float, default=50.0,
                    help='reference solve rate that defines the discriminating point')
    args = ap.parse_args(argv)

    data = collect_solve_episodes(Path(args.runs_dir))
    if not data:
        raise SystemExit('no bdm_* runs found')

    pooled = defaultdict(list)
    for cell in data.values():
        for a, eps in cell.items():
            pooled[a] += eps

    grid = np.unique(np.concatenate([np.arange(25, 1000, 25), np.arange(1000, 4001, 50)]))
    curves = {a: np.array([rate(pooled[a], b) for b in grid]) for a in ARMS if a in pooled}

    # Evaluation budget chosen from the REFERENCE arm alone — never from the CCE arms.
    ref = curves[args.reference]
    e_star = int(grid[int(np.argmin(np.abs(ref - args.target_rate)))])

    spread = np.array([max(curves[a][i] for a in curves) - min(curves[a][i] for a in curves)
                       for i in range(len(grid))])

    print(f"reference arm '{args.reference}' hits {args.target_rate:.0f}% solved at "
          f"episode {e_star}")
    print(f"peak arm-to-arm spread {spread.max():.1f}pp at episode {int(grid[spread.argmax()])}")
    print(f"spread at the budget we actually used (4000): {spread[-1]:.1f}pp\n")
    print(f"{'arm':<14} {'@' + str(e_star):>9} {'@4000':>8}")
    for a in ARMS:
        if a in curves:
            print(f"{LABEL[a]:<14} {rate(pooled[a], e_star):>8.0f}% {rate(pooled[a], 4000):>7.0f}%")

    _plot(grid, curves, spread, e_star, data, args)
    return e_star


def _plot(grid, curves, spread, e_star, data, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.28)

    # (a) solve-rate curves
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = 200, 1000
    ax.axvspan(lo, hi, color='#B4600F', alpha=0.10, lw=0)
    ax.text((lo + hi) / 2, 4, 'discriminating\nwindow', ha='center', va='bottom',
            fontsize=8.5, color='#B4600F')
    for a, c in curves.items():
        ax.plot(grid, c, color=COLOR[a], lw=1.9, label=LABEL[a])
    ax.axvline(4000, color='#9E2F27', ls='--', lw=1.3)
    ax.annotate('we measured here\n(every arm at 100%)', xy=(4000, 100),
                xytext=(1500, 72), fontsize=8.5, color='#9E2F27',
                arrowprops=dict(arrowstyle='->', color='#9E2F27', lw=1.1))
    ax.axvline(e_star, color='#122236', ls=':', lw=1.4)
    ax.text(e_star + 60, 30, f'E*={e_star}', fontsize=8.5, color='#122236')
    ax.set_xlabel('training budget (episodes)')
    ax.set_ylabel('% of routing seeds reaching the exact optimum')
    ax.set_title('(a) The routing metric saturates long before we look',
                 fontsize=11, loc='left')
    ax.set_ylim(0, 104); ax.set_xlim(0, 4050)
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc='center right')

    # (b) measurement power
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(grid, spread, color='#B4600F', alpha=.22, lw=0)
    ax.plot(grid, spread, color='#B4600F', lw=1.9)
    ax.axvline(4000, color='#9E2F27', ls='--', lw=1.3)
    ax.set_xlabel('training budget (episodes)')
    ax.set_ylabel('best arm − worst arm  (percentage points)')
    ax.set_title('(b) How much difference is even visible', fontsize=11, loc='left')
    ax.set_xlim(0, 4050); ax.grid(alpha=.25)
    ax.annotate(f'{spread[-1]:.0f}pp', xy=(4000, spread[-1]), xytext=(2900, spread.max() * .45),
                fontsize=9, color='#9E2F27',
                arrowprops=dict(arrowstyle='->', color='#9E2F27', lw=1.1))

    # (c) per-cell solve rate at E*
    ax = fig.add_subplot(gs[0, 2])
    cells = sorted(data, key=lambda k: -(max(rate(data[k][a], e_star) for a in ARMS if a in data[k])
                                         - min(rate(data[k][a], e_star) for a in ARMS if a in data[k])))[:4]
    width = 0.16
    xs = np.arange(len(cells))
    for i, a in enumerate(ARMS):
        vals = [rate(data[c].get(a, []), e_star) for c in cells]
        ax.bar(xs + (i - 2) * width, vals, width, color=COLOR[a], label=LABEL[a])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"B={c[0]:.2f}\ncap {c[1]}" for c in cells], fontsize=8.5)
    ax.set_ylabel(f'% seeds solved by episode {e_star}')
    ax.set_title(f'(c) Where arms actually separate (E={e_star})', fontsize=11, loc='left')
    ax.grid(alpha=.25, axis='y'); ax.set_ylim(0, 100)

    # Be explicit about WHOSE data this is: the metric is borrowed from the FrozenLake
    # analysis, but every number here is budget routing. An earlier title said
    # "the metric FrozenLake used", which read as if the data were FrozenLake's.
    n_runs = sum(len(v) for cell in data.values() for v in cell.values())
    fig.suptitle('BUDGET ROUTING (CVRP) — measured after the race was already over',
                 fontsize=13, y=1.045)
    fig.text(0.5, 0.995,
             f'{n_runs} routing runs, {len(data)} cells, 12 seeds per arm · '
             f'no FrozenLake data in this figure — only its per-seed solve-rate metric',
             ha='center', fontsize=9.5, color='#48586B')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_c2_cvrp_solve_rate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"\nwrote {p}")


if __name__ == '__main__':
    main()
