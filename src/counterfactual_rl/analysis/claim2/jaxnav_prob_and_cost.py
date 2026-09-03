"""The two analyses Jeremy asked for that need NO new compute (07/10 and 08/13).

Both read the finished bootstrap sweep (array 274476, 50/50 COMPLETED) off disk.

1. P(METHOD > BASELINE), Jeremy 08/13: "even 80-90% would be a strong, reportable
   result", and "one-tailed uses 90% confidence; two-tailed uses 95%". We have been
   handing him two-tailed p-values instead, which answer a different question. This
   reports rliable's probability of improvement -- P(a random seed of the method beats
   a random seed of the baseline) -- with BOTH interval widths, so the one-tailed 90%
   reading he asked for is present and labelled.

   Reported against BOTH baselines on purpose. PER is a weak baseline on JaxNav and
   gets weaker with clutter (win rate 75 -> 31 -> 18 as fill goes 0.1 -> 0.3 -> 0.5,
   while uniform degrades 73 -> 60 -> 49), so "beats PER" overstates the result.
   UNIFORM is the honest bar and is printed first.

2. COMPUTE-TIME ANALYSIS, Jeremy 07/10, never done. Every run already writes
   timing.jsonl; the 'total' entry is ground-truth wall clock and the
   update.scoring.* leaves are the CCE overhead. Reports hours per run and the
   scoring share, i.e. what the method costs for whatever it buys.

Reuses analysis.claim2.compute_metrics.prob_improvement and .parse_wallclock rather
than reimplementing either.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_prob_and_cost [--array 274476]
"""
import argparse
import json
import os

import numpy as np

from rliable import library as rly
from rliable import metrics as rl_metrics

from .compute_metrics import parse_wallclock
from .jaxnav_bootstrap_curves import MANIFEST, ROOT, RUNS, resolve

# Last-20-eval mean per seed: the summary the lab notebook uses everywhere else for
# JaxNav, so these numbers are comparable with every table already written down.
N_FINAL_EVALS = 20


def load_final(array_id):
    """(arm) -> (final_scores (n_seeds,), run_dirs [str]). COMPLETED runs only."""
    man = json.load(open(MANIFEST))
    resolved = resolve(array_id)
    finals, dirs = {}, {}
    for rec in man.values():
        jid = rec["job_id"]
        if jid not in resolved:
            continue
        run_dir, state = resolved[jid]
        if state != "COMPLETED":
            continue
        f = os.path.join(RUNS, run_dir, "metrics.log")
        if not os.path.exists(f):
            continue
        wr = []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 6 or p[0] == "episode":
                continue
            wr.append(float(p[3].rstrip("%")))
        if len(wr) < N_FINAL_EVALS:
            continue
        finals.setdefault(rec["arm"], []).append(float(np.mean(wr[-N_FINAL_EVALS:])))
        dirs.setdefault(rec["arm"], []).append(os.path.join(RUNS, run_dir))
    return {k: np.array(v) for k, v in finals.items()}, dirs


def _prob(finals, baseline, ci_size, reps):
    """P(arm > baseline) with an explicit interval width.

    compute_metrics.prob_improvement does the same thing but hard-codes rliable's
    default 95% interval, and Jeremy 08/13 asked specifically for the one-tailed
    90% reading, so the width has to be a parameter here.
    """
    base = finals[baseline].reshape(-1, 1)
    # One call for ALL arms: get_interval_estimates takes a dict and resamples once,
    # which is both the intended API and ~9x less work than a call per arm.
    scores = {a: v.reshape(-1, 1) for a, v in finals.items() if a != baseline}
    fn = lambda x, y=base: rl_metrics.probability_of_improvement(x, y)
    point, ci = rly.get_interval_estimates(
        scores, fn, reps=reps, confidence_interval_size=ci_size)
    return {a: (float(np.squeeze(point[a])),
                float(np.squeeze(ci[a][0])),
                float(np.squeeze(ci[a][1]))) for a in scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274476")
    ap.add_argument("--reps", type=int, default=50000)
    args = ap.parse_args()

    finals, dirs = load_final(args.array)
    if not finals:
        print("no COMPLETED runs resolved -- is the array id right?")
        return

    print(f"\n{'='*74}\n  JaxNav 8x8 fill 0.3 -- array {args.array}, final win % "
          f"(mean of last {N_FINAL_EVALS} evals per seed)\n{'='*74}")
    for arm in sorted(finals):
        v = finals[arm]
        print(f"  {arm:<16} n={len(v)}  median {np.median(v):5.1f}   "
              f"seeds {np.sort(v).round(1)}")

    # ---- 1. P(method > baseline) -------------------------------------------
    for baseline in ("uniform", "per"):
        if baseline not in finals:
            continue
        note = ("the honest bar" if baseline == "uniform"
                else "WEAK on JaxNav -- read with the uniform column")
        print(f"\n{'-'*78}\n  P(arm > {baseline})   [{note}]\n{'-'*78}")
        print(f"  {'arm':<16} {'P(better)':>10}   {'90% interval':>20}"
              f"   {'95% interval':>20}")
        r90 = _prob(finals, baseline, 0.90, args.reps)
        r95 = _prob(finals, baseline, 0.95, args.reps)
        for arm in sorted(r90):
            p, lo, hi = r90[arm]
            _, lo95, hi95 = r95[arm]
            flag = "  <-" if lo >= 0.5 else ""
            print(f"  {arm:<16} {p:>10.3f}   {f'[{lo:.3f}, {hi:.3f}]':>20}"
                  f"   {f'[{lo95:.3f}, {hi95:.3f}]':>20}{flag}")
        print(f"\n  n={len(finals[baseline])} seeds per arm. Jeremy 08/13: one-tailed reads"
              f" the 90% column,\n  two-tailed the 95%. An arm is marked <- only when the"
              f" 90% interval CLEARS 0.5;\n  a bare point estimate above 0.8 with an"
              f" interval spanning 0.5 is not a result.\n  With 5 seeds these intervals are"
              f" very wide -- ~0.5 means 'no signal', not 'no difference'.")

    # ---- 2. compute cost ----------------------------------------------------
    wc = parse_wallclock(dirs)
    print(f"\n{'='*78}\n  COMPUTE COST per run (timing.jsonl 'total' = ground truth)"
          f"\n{'='*78}")
    print(f"  {'arm':<16} {'total h':>9} {'vs uniform':>11} {'n runs':>8}")
    base_h = wc.get("uniform", {}).get("total_hours")
    for arm in sorted(wc):
        d = wc[arm]
        rel = f"{d['total_hours']/base_h:.1f}x" if base_h else "n/a"
        print(f"  {arm:<16} {d['total_hours']:>9.2f} {rel:>11} {len(dirs[arm]):>8}")

    # NO scoring/training split. parse_wallclock derives it from update.scoring.*
    # leaf timers, and the JaxNav trainer instruments only 'total' and 'eval' -- so
    # the split comes back 0.00 for every arm and "training_hours == total_hours" is
    # an artefact of the missing timers, not a measurement. Reporting it would say
    # "scoring costs nothing", which is the opposite of the truth.
    scoring = sum(d["scoring_hours"] for d in wc.values())
    if scoring == 0.0:
        print("\n  NOT AVAILABLE: the scoring-vs-training split. parse_wallclock reads it")
        print("  from update.scoring.* timers, and the JaxNav trainer records only 'total'")
        print("  and 'eval'. The totals above are real; the split needs timers added to")
        print("  consequence_dqn._score_buffer_transitions first. Do not infer it from the")
        print("  CCE-minus-uniform gap either -- that difference also contains the extra")
        print("  buffer traffic and the rollout compile.")
    print("\n  Read next to P(better) above: this is what CCE costs for what it buys.")
    print()


if __name__ == "__main__":
    main()
