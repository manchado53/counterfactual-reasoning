"""Analysis for the ESS-matched balance sweep on deterministic FrozenLake.

FL-det is BIMODAL: verified on the paper's own cache
(`paper/repro/cache/claim2_frozen_lake_no_slip.npz`), zero of 124 seeds ever sit
between 0.1 and 0.9 -- a seed either bootstraps the goal reward back to the start
and ends at ~1.0, or never catches the thread and ends at 0.0.

So FRACTION OF SEEDS SOLVED is the primary metric, not IQM. The paper's headline
"80% vs 48%" is frac-solved; the "1.00 vs 0.46" in the figure is IQM, which trims
the dead seeds out of the middle and reads higher. Both are reported here so the
two can never be confused again.

Because the outcome is binary and the x-axis is ordered, the trend test is
Cochran-Armitage, not Spearman.

    python -m counterfactual_rl.analysis.claim2.fl_balance_ess
"""
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                    "docs/figures/real/claim2/frozen_lake/data")
RUNS = os.path.join(os.path.dirname(__file__), "..", "..",
                    "agents", "frozen_lake", "runs")
MANIFEST = os.path.join(DATA, "manifest_fl_balance_ess.json")
SOLVED_THRESHOLD = 0.5     # bimodal, so any cut in (0.1, 0.9) gives the same answer


def _arms():
    with open(MANIFEST) as fh:
        man = json.load(fh)
    out = {}
    for rec in man.values():
        out.setdefault(float(rec["cce_balance"]), []).append(str(rec["job_id"]))
    return dict(sorted(out.items()))


def _final_win(job, k=3):
    """Mean win rate over the last k evals, or None if the run produced none."""
    f = os.path.join(RUNS, str(job), "metrics.log")
    if not os.path.exists(f):
        return None
    vals = []
    for line in open(f):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4 or parts[0] == "episode":
            continue
        try:
            vals.append(float(parts[3].rstrip("%")) / 100.0)
        except ValueError:
            continue
    return float(np.mean(vals[-k:])) if vals else None


def _ess(job):
    p = os.path.join(RUNS, str(job), "ess.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _wilson(k, n, z=1.96):
    """Wilson score interval -- correct for proportions near 0 or 1, unlike normal-approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _cochran_armitage(counts):
    """Trend test for a binary outcome across ORDERED doses.

    counts: list of (dose, n_solved, n_total). Returns (z, two-sided p).
    """
    d = np.array([c[0] for c in counts], float)
    x = np.array([c[1] for c in counts], float)
    n = np.array([c[2] for c in counts], float)
    N, X = n.sum(), x.sum()
    if N == 0 or X in (0, N):
        return 0.0, 1.0
    p = X / N
    dbar = (n * d).sum() / N
    num = (x * (d - dbar)).sum()
    var = p * (1 - p) * (n * (d - dbar) ** 2).sum()
    if var <= 0:
        return 0.0, 1.0
    z = num / np.sqrt(var)
    from scipy.stats import norm
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def main():
    arms = _arms()
    print(f"loaded {sum(len(v) for v in arms.values())} runs, {len(arms)} balance levels")

    print("\n=== 1. did ESS matching hold? (target 0.60) ===")
    print(f"{'balance':>8} {'runs w/ data':>13} {'ess_frac med':>13} {'saturated':>11}")
    ok = True
    for bal, jobs in arms.items():
        vals = [r["ess_frac"] for j in jobs for r in _ess(j) if not r.get("ess_k_saturated")]
        sat = sum(1 for j in jobs for r in _ess(j) if r.get("ess_k_saturated"))
        withdata = sum(1 for j in jobs if _ess(j))
        if not vals:
            print(f"{bal:>8} {withdata:>13}   NO DATA")
            ok = False
            continue
        med = float(np.median(vals))
        flag = "" if abs(med - 0.6) <= 0.05 else "  <-- OFF TARGET"
        ok &= not flag
        print(f"{bal:>8} {withdata:>13} {med:>13.3f} {sat:>11}{flag}")
    print("VERDICT:", "matched" if ok else "NOT MATCHED -- trend below is confounded")

    print("\n=== 2. fraction of seeds SOLVED (the paper's headline metric) ===")
    print(f"{'balance':>8} {'solved':>9} {'frac':>7} {'95% CI (Wilson)':>20} {'IQM':>7}")
    from scipy.stats import trim_mean
    counts, per = [], {}
    for bal, jobs in arms.items():
        fins = [v for v in (_final_win(j) for j in jobs) if v is not None]
        if not fins:
            print(f"{bal:>8}   no finished runs")
            continue
        a = np.array(fins)
        k, n = int((a > SOLVED_THRESHOLD).sum()), len(a)
        lo, hi = _wilson(k, n)
        counts.append((bal, k, n))
        per[bal] = a
        print(f"{bal:>8} {f'{k}/{n}':>9} {k/n:>7.2f} {f'[{lo:.2f},{hi:.2f}]':>20} "
              f"{trim_mean(a, 0.25):>7.3f}")
        mid = int(((a > 0.1) & (a < 0.9)).sum())
        if mid:
            print(f"{'':>8}   note: {mid} middling seed(s) -- bimodality assumption weakened")

    if len(counts) >= 3:
        z, p = _cochran_armitage(counts)
        print(f"\nTREND (Cochran-Armitage, binary outcome over ordered balance):"
              f"  z = {z:+.3f}   p = {p:.4f}")
        print("  z > 0 -> more CCE solves more seeds;  z ~ 0 -> balance does not matter")

    if 0.0 in per:
        print("\n=== 3. each arm vs balance 0.0 (= PER at matched concentration) ===")
        from scipy.stats import fisher_exact
        k0 = int((per[0.0] > SOLVED_THRESHOLD).sum()); n0 = len(per[0.0])
        print(f"{'balance':>8} {'solved':>9} {'Fisher p vs PER':>17}")
        for bal, a in per.items():
            if bal == 0.0:
                continue
            k, n = int((a > SOLVED_THRESHOLD).sum()), len(a)
            _, pf = fisher_exact([[k, n - k], [k0, n0 - k0]])
            print(f"{bal:>8} {f'{k}/{n}':>9} {pf:>17.4f}")
        print(f"\n  reference: paper reports CCE-mul 20/25 (80%) vs PER 12/25 (48%) "
              f"at beta=0.25, mu=1/1 (UNMATCHED concentration)")


if __name__ == "__main__":
    main()
