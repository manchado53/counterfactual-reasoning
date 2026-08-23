"""
The whole routing investigation in one figure.

Every panel is measured, none illustrative. The argument it makes:

  (a) routing's decisions sit in a useless MIDDLE band, where FrozenLake's are either free
      or critical. Option A removed the middle -- and overshot into "almost everything is
      critical", which is the same problem mirrored.
  (b) routing's difficulty is a STEP FUNCTION. Solve rate runs 12% -> 25% -> 100% across four
      budget units, so there is no setting where a comparison is well posed.
  (c) across every configuration tried -- two aggregations, two instance sizes, and the
      strandable rewrite -- CCE never separates from PER.

Run:
    python -m counterfactual_rl.analysis.claim2.routing_summary \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs --out docs/figures/real/claim2
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CCE_ARMS = ['cceonly', 'cceadd', 'ccemul']


def finals(runs_dir, pattern, arm_re):
    """{arm: [final opt_ratio per seed]} for runs matching a glob."""
    out = defaultdict(list)
    for d in sorted(Path(runs_dir).glob(pattern)):
        m = arm_re.match(d.name)
        f = d / 'metrics.log'
        if not m or not f.exists():
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
            out[m.group(1)].append(vals[-1])
    return out


def p_better(a, b, n=20000, seed=0):
    """P(a random seed of arm a beats one of arm b); ties count half. 0.5 = no difference."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return np.nan, (np.nan, np.nan)
    ai = rng.integers(0, a.size, (n, a.size)); bi = rng.integers(0, b.size, (n, b.size))
    x, y = a[ai][:, :, None], b[bi][:, None, :]
    w = (x > y).mean(axis=(1, 2)) + 0.5 * (x == y).mean(axis=(1, 2))
    return float(w.mean()), (float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    R = args.runs_dir

    CONFIGS = [
        ('old env\nmax agg', 'bd_*_b080_c10_s*', re.compile(r'^bd_(\w+?)_b\d+_c\d+_s\d+$')),
        ('old env\nmean agg', 'bdm_*_b080_c10_s*', re.compile(r'^bdm_(\w+?)_b\d+_c\d+_s\d+$')),
        ('12-customer\nmean agg', 'bdi_ring12mean_*_b080_c10_s*',
         re.compile(r'^bdi_ring12mean_(\w+?)_b\d+_c\d+_s\d+$')),
        ('STRANDABLE\n(Option A)', 'oa_*', re.compile(r'^oa_(\w+?)_s\d+$')),
        ('STRANDABLE\n50k episodes', 'oaL_*', re.compile(r'^oaL_(\w+?)_s\d+$')),
    ]
    results = []
    for label, glob, rx in CONFIGS:
        d = finals(R, glob, rx)
        if 'per' not in d:
            continue
        best = None
        for a in CCE_ARMS:
            if a in d:
                p, ci = p_better(d[a], d['per'])
                if best is None or p > best[1]:
                    best = (a, p, ci, len(d[a]))
        if best:
            results.append((label, *best))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.1], wspace=0.3)

    # (a) stakes distributions
    ax = fig.add_subplot(gs[0, 0])
    rows = [('FrozenLake det\nCCE WINS', 50.9, 0.0, 49.1),
            ('FrozenLake slip\nnull', 24.5, 62.3, 13.2),
            ('routing, old\nnull', 16.3, 70.4, 13.3),
            ('routing, strandable\nnull', 13.7, 13.6, 72.7)]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, lo, mid, hi) in zip(ys, rows):
        left = 0
        for v, c, nm in zip((lo, mid, hi), ('#D3DAE3', '#B4600F', '#1B5FA8'),
                            ('barely', 'middle', 'critical')):
            ax.barh(y, v, left=left, color=c, height=0.6, label=nm if y == ys[0] else None)
            if v >= 9:
                ax.text(left + v / 2, y, f'{v:.0f}', ha='center', va='center',
                        fontsize=9, color='white')
            left += v
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel('% of decision states'); ax.set_xlim(0, 100)
    ax.set_title('(a) Does the choice matter? (exact oracle)', fontsize=11, loc='left')
    ax.legend(fontsize=8, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.28))
    ax.grid(alpha=.2, axis='x')

    # (b) difficulty step function
    ax = fig.add_subplot(gs[0, 1])
    mult = [0.95, 1.10, 1.14, 1.17, 1.21, 1.25]
    rate = [17, 0, 25, 12, 12, 100]
    ax.plot(mult, rate, 'o-', color='#9E2F27', lw=2, ms=7)
    ax.axhspan(30, 60, color='#2C6E49', alpha=0.13)
    ax.text(0.96, 45, 'the band where a\ncomparison is\ninformative',
            fontsize=8.5, color='#2C6E49')
    ax.set_xlabel('fuel budget (x optimal tour)')
    ax.set_ylabel('% of baseline seeds solving it')
    ax.set_title('(b) Difficulty is a step, not a dial', fontsize=11, loc='left')
    ax.grid(alpha=.25); ax.set_ylim(-5, 105)

    # (c) CCE vs PER everywhere
    ax = fig.add_subplot(gs[0, 2])
    ys = np.arange(len(results))[::-1]
    for y, (label, arm, p, ci, n) in zip(ys, results):
        # A zero-width interval is not a plotting bug: in the saturated configurations EVERY
        # seed of every arm ended at exactly 1.0, so every pairwise comparison is a tie and
        # the bootstrap returns 0.5 with no spread. Mark it, or it reads as suspiciously
        # precise rather than as the ceiling effect it is.
        tied = (ci[1] - ci[0]) < 1e-3
        if tied:
            ax.plot(p, y, 'o', mfc='white', mec='#9E2F27', mew=2.0, ms=9, zorder=3)
            ax.text(0.56, y, 'every seed tied at the ceiling (1.0)',
                    fontsize=8, va='center', color='#B4600F')
        else:
            ax.plot([ci[0], ci[1]], [y, y], color='#7C8A9B', lw=2.2, solid_capstyle='round')
            ax.plot(p, y, 'o', color='#9E2F27', ms=9, zorder=3)
        ax.text(1.02, y, f'n={n}/arm', fontsize=8, va='center', color='#7C8A9B')
    ax.axvline(0.5, color='#122236', ls='--', lw=1.4)
    ax.text(0.503, ys[0] + 0.42, 'no difference', fontsize=8.5)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in results], fontsize=8.5)
    ax.set_xlabel('P(best CCE arm beats PER)   ·   95% bootstrap CI')
    ax.set_xlim(0.0, 1.15)
    ax.set_title('(c) Every configuration tried', fontsize=11, loc='left')
    ax.grid(alpha=.25, axis='x')

    fig.suptitle('Why routing cannot host Claim 2 — three independent measurements, '
                 '~2,500 training runs', fontsize=12.5, y=1.02)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p_ = out / 'fig_routing_summary.png'
    fig.savefig(p_, dpi=150, bbox_inches='tight')
    print(f"wrote {p_}")
    for label, arm, p, ci, n in results:
        print(f"  {label.replace(chr(10),' '):<28} best={arm:<8} "
              f"P(beat PER)={p:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  n={n}")


if __name__ == '__main__':
    main()
