"""Analysis for the ESS-matched balance sweep (slurm/sweep_balance_ess.py).

The sweep varies one thing: `cce_balance`, the share of a FIXED replay
concentration that is driven by the CCE score rather than the TD error. So the
question this answers is narrow and clean:

    at equal sampler concentration, is the CCE signal better than the TD signal?

Three things are reported, in the order they should be read:

1. `verify_ess` -- did the matching actually hold? Every claim below is void if
   the arms ran at different concentrations, so this is checked FIRST and
   failure is printed loudly rather than folded into a footnote. Also surfaces
   `ess_k_saturated`, the flag for evals where the target was unreachable
   because the driving signal was degenerate.
2. `dose_response` -- IQM final win rate across the balance axis with bootstrap
   CIs, plus a Spearman trend test over the axis. The TREND is the claim; the
   five points are not five independent pairwise tests.
3. `prob_better` -- bootstrap P(arm > balance 0.0), the statistic the advisor
   asked for on 2026-08-13 (a one-sided empirical probability, where 80-90% is
   reportable, rather than a two-tailed significance test).

Usage:
    python -m counterfactual_rl.analysis.claim2.jaxnav_balance_ess
"""
import json
import os

import numpy as np

from .jaxnav_holes_figures import DATA, finals, load  # noqa: F401
from .compute_metrics import iqm

MANIFEST = os.path.join(DATA, "manifest_balance_ess.json")
RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "agents", "jax_nav", "runs")
N_BOOT = 20000


def _manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def _by_balance(man):
    out = {}
    for rec in man.values():
        out.setdefault(float(rec["cce_balance"]), []).append(rec["job_id"])
    return dict(sorted(out.items()))


def _ess_records(job):
    path = os.path.join(RUNS, str(job), "ess.jsonl")
    if not os.path.exists(path):
        return []
    recs = []
    with open(path) as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def verify_ess(arms, target=0.6, tol=0.05):
    """Check the concentration matching held. Returns True only if every arm is on target."""
    print("\n=== 1. did ESS matching hold? (target %.2f +/- %.2f) ===" % (target, tol))
    print(f"{'balance':>8} {'runs':>5} {'ess_frac med':>13} {'IQR':>16} {'saturated evals':>16}")
    ok = True
    for bal, jobs in arms.items():
        vals, sat, n = [], 0, 0
        for j in jobs:
            for r in _ess_records(j):
                # ignore the warmup evals before the first scoring pass, where a
                # pure-CCE arm has no signal to concentrate on yet
                if r.get("ess_k_saturated"):
                    sat += 1
                else:
                    vals.append(r["ess_frac"])
                n += 1
        if not vals:
            print(f"{bal:>8} {len(jobs):>5}   NO DATA")
            ok = False
            continue
        med = float(np.median(vals))
        lo, hi = np.percentile(vals, [25, 75])
        flag = "" if abs(med - target) <= tol else "   <-- OFF TARGET"
        if flag:
            ok = False
        pct = 100.0 * sat / max(n, 1)
        print(f"{bal:>8} {len(jobs):>5} {med:>13.3f} {f'[{lo:.3f},{hi:.3f}]':>16} "
              f"{f'{sat} ({pct:.0f}%)':>16}{flag}")
    print("VERDICT:", "matched -- comparisons below are valid"
          if ok else "NOT MATCHED -- the sweep is confounded, do not read the trend as signal quality")
    return ok


def _boot_iqm(vals, rng):
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    return np.array([iqm(np.asarray(vals)[i]) for i in idx])


def dose_response(arms, k=5):
    print("\n=== 2. dose-response: final win rate vs balance ===")
    print(f"{'balance':>8} {'n':>3} {'IQM':>8} {'95% CI':>18} {'mean':>8}")
    rng = np.random.default_rng(0)
    xs, ys, per_arm = [], [], {}
    for bal, jobs in arms.items():
        vals = finals(jobs, k=k)
        vals = np.asarray([v for v in vals if np.isfinite(v)])
        if len(vals) == 0:
            print(f"{bal:>8}   no finished runs")
            continue
        b = _boot_iqm(vals, rng)
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"{bal:>8} {len(vals):>3} {iqm(vals):>8.3f} {f'[{lo:.3f},{hi:.3f}]':>18} {vals.mean():>8.3f}")
        per_arm[bal] = vals
        xs.extend([bal] * len(vals))
        ys.extend(vals.tolist())

    if len(set(xs)) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        boot = []
        xs_a, ys_a = np.asarray(xs), np.asarray(ys)
        for _ in range(2000):
            i = rng.integers(0, len(xs_a), len(xs_a))
            if len(set(xs_a[i])) < 2:
                continue
            boot.append(spearmanr(xs_a[i], ys_a[i]).statistic)
        blo, bhi = np.percentile(boot, [2.5, 97.5])
        print(f"\nTREND across the axis: Spearman rho = {rho:+.3f}  p = {p:.4f}"
              f"  bootstrap 95% CI [{blo:+.3f},{bhi:+.3f}]")
        print("  rho > 0 -> more CCE is better;  rho ~ 0 -> CCE adds nothing;"
              "  rho < 0 -> TD is the better signal")
    return per_arm


def prob_better(per_arm, ref=0.0):
    print(f"\n=== 3. bootstrap P(arm > balance {ref}) ===")
    if ref not in per_arm:
        print(f"  reference arm {ref} has no data")
        return
    rng = np.random.default_rng(1)
    base = per_arm[ref]
    print(f"{'balance':>8} {'P(>ref)':>9}   (advisor 2026-08-13: 80-90% is reportable)")
    for bal, vals in per_arm.items():
        if bal == ref:
            continue
        a = rng.choice(vals, size=(N_BOOT, len(vals)))
        b = rng.choice(base, size=(N_BOOT, len(base)))
        p = float(np.mean(a.mean(1) > b.mean(1)))
        print(f"{bal:>8} {p:>9.3f}")


def main():
    man = _manifest()
    arms = _by_balance(man)
    print(f"loaded {len(man)} runs across {len(arms)} balance levels")
    matched = verify_ess(arms)
    per_arm = dose_response(arms)
    prob_better(per_arm)
    if not matched:
        print("\nREMINDER: ESS matching failed above. Treat the trend as descriptive only.")


if __name__ == "__main__":
    main()
