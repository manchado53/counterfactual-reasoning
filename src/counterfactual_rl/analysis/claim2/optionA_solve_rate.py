"""
Option A sweep — per-seed SOLVE RATE, the FrozenLake metric.

Written BEFORE the sweep finished, so the test is not shaped by the numbers.

The configuration and the evaluation budget E* were fixed in the pre-registration commit
(lab-notebook, 2026-08-20) from the DQN-uniform curve alone. The registered prediction was the
PESSIMISTIC one: CCE will NOT beat PER by a FrozenLake-sized margin (~30pp), because Gate 1
measured 72.7% of this configuration's decisions as critical against FrozenLake's 49.1% -- a
ranking cannot isolate a small important set when most states are important.

Reported per arm:
  solved      seeds whose FINAL greedy policy matches the exact oracle
  95% CI      Wilson interval (correct for proportions near 0 or 1, unlike normal approx)
  vs PER      difference in percentage points, with a Fisher exact two-sided p

Run:
    python -m counterfactual_rl.analysis.claim2.optionA_solve_rate \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ['uniform', 'per', 'cceonly', 'cceadd', 'ccemul']
LABEL = {'uniform': 'DQN-Uniform', 'per': 'DQN+PER', 'cceonly': 'DQN+CCE-only',
         'cceadd': 'CCE+TD (add)', 'ccemul': 'CCE+TD (mul)'}
RUN_RE = re.compile(r'^oa_(\w+?)_s(\d+)$')
REGISTERED_MARGIN = 30.0        # pp; the FrozenLake-sized effect the prediction is about


def final_score(path: Path):
    vals = []
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        if len(f) >= 4:
            try:
                vals.append(float(f[3]))
            except ValueError:
                pass
    return vals[-1] if vals else None


def wilson(k, n, z=1.96):
    """Wilson score interval — behaves sensibly at 0/n and n/n, unlike the normal approx."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def fisher_p(k1, n1, k2, n2):
    """Two-sided Fisher exact test, via scipy if present, else a hypergeometric fallback."""
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    except Exception:
        from math import comb
        tot, succ = n1 + n2, k1 + k2
        def pmf(k):
            return comb(n1, k) * comb(n2, succ - k) / comb(tot, succ)
        obs = pmf(k1)
        lo = max(0, succ - n2)
        return float(sum(pmf(k) for k in range(lo, min(n1, succ) + 1)
                         if pmf(k) <= obs + 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--threshold', type=float, default=0.9999,
                    help='opt_ratio counting as solved (default: exactly optimal)')
    args = ap.parse_args(argv)

    scores = defaultdict(list)
    for d in sorted(Path(args.runs_dir).glob('oa_*')):
        m = RUN_RE.match(d.name)
        f = d / 'metrics.log'
        if not m or not f.exists():
            continue
        v = final_score(f)
        if v is not None:
            scores[m.group(1)].append(v)

    if 'per' not in scores:
        raise SystemExit('no PER runs found — cannot compare')

    pk = sum(1 for v in scores['per'] if v >= args.threshold)
    pn = len(scores['per'])

    print("\nOPTION A — strandable budget routing, pre-registered at E*=5500\n")
    print(f"{'arm':<15} {'n':>4} {'solved':>12} {'95% CI':>16} {'vs PER':>9} {'p':>8}")
    print("-" * 72)
    rows = {}
    for a in ARMS:
        if a not in scores:
            continue
        v = np.array(scores[a]); n = len(v)
        k = int((v >= args.threshold).sum())
        lo, hi = wilson(k, n)
        rate = 100 * k / n
        if a == 'per':
            print(f"{LABEL[a]:<15} {n:>4} {k:>4}/{n:<3} {rate:>4.0f}% "
                  f"[{lo:>5.1f},{hi:>5.1f}] {'—':>9} {'—':>8}")
        else:
            diff = rate - 100 * pk / pn
            p = fisher_p(k, n, pk, pn)
            print(f"{LABEL[a]:<15} {n:>4} {k:>4}/{n:<3} {rate:>4.0f}% "
                  f"[{lo:>5.1f},{hi:>5.1f}] {diff:>+8.1f}pp {p:>8.3f}")
        rows[a] = dict(n=n, solved=k, rate=rate)

    print()
    cce = {a: r for a, r in rows.items() if a.startswith('cce')}
    if cce:
        best = max(cce.values(), key=lambda r: r['rate'])
        best_a = [a for a, r in cce.items() if r is best][0]
        gap = best['rate'] - 100 * pk / pn
        print(f"best CCE arm: {LABEL[best_a]} at {best['rate']:.0f}% "
              f"({gap:+.1f}pp vs PER)")
        if gap >= REGISTERED_MARGIN:
            print(f"REGISTERED PREDICTION REFUTED — a >={REGISTERED_MARGIN:.0f}pp win. The "
                  f"'needs a MINORITY of critical states' clause is wrong.")
        elif gap > 0:
            print(f"Prediction HELD in direction: a win, but under the {REGISTERED_MARGIN:.0f}pp "
                  f"FrozenLake-sized margin the prediction was about.")
        else:
            print("Prediction HELD — no CCE arm beats PER.")
    print("\nNote: 40 seeds resolves roughly a 30pp difference. A true effect near 10pp is "
          "invisible at this sample size and must be reported as a limit, not as absence.")
    return rows


if __name__ == '__main__':
    main()
