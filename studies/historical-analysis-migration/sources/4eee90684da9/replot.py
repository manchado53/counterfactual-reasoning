"""Regenerate the paper's Claim-2 figures from the cached arrays ALONE.

No raw run directories are touched -- this reads only paper/repro/cache/*.npz,
which is the whole point: the figures survive even if the ~260 GB of runs are
purged.

Run from the repo root:
    PYTHONPATH=src python paper/repro/replot.py            # quick check (reps=2000)
    PYTHONPATH=src python paper/repro/replot.py --reps 50000   # exact paper match

Outputs -> paper/repro/regen/  (compare against paper/figures/)
"""
import argparse
import json
import os

import numpy as np

from counterfactual_rl.analysis.claim2.compute_metrics import compute_all
from counterfactual_rl.analysis.claim2.plot_figures import (
    fig1_iqm_curves,
    fig2_final_iqm,
    fig4b_prob_improvement_curves,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "regen")

CACHES = ["claim2_frozen_lake_no_slip", "claim2_frozen_lake", "claim2_smax_3m"]


def load_cache(name):
    d = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    labels = meta["labels"]
    raw = {a: d[f"raw_{i}"] for i, a in enumerate(labels)}
    eval_steps = {a: d[f"steps_{i}"] for i, a in enumerate(labels)}
    raw_length = {a: d[f"len_{i}"] for i, a in enumerate(labels) if f"len_{i}" in d}
    return meta, raw, eval_steps, (raw_length or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000,
                    help="bootstrap resamples (50000 reproduces the paper exactly)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for name in CACHES:
        meta, raw, eval_steps, raw_length = load_cache(name)
        env = meta["env"]
        threshold = meta["threshold"]
        env_name = env.replace("_", " ").title()
        print(f"\n=== {name}  (env={env}, threshold={threshold}, reps={args.reps}) ===")

        results = compute_all(raw, eval_steps, threshold,
                              raw_length=raw_length, run_dirs=None, reps=args.reps)

        print("Final IQM:")
        for a, (pt, lo, hi) in results["final_iqm"].items():
            print(f"  {a:<14} {pt:.3f}  [{lo:.3f}, {hi:.3f}]")

        steps0 = next(iter(eval_steps.values()))
        fig1_iqm_curves({env_name: results["iqm_curves"]}, {env_name: steps0},
                        {env_name: threshold},
                        os.path.join(OUT, f"fig1_iqm_{env}.png"))
        fig2_final_iqm({env_name: results["final_iqm"]},
                       os.path.join(OUT, f"fig2_final_iqm_{env}.png"))
        fig4b_prob_improvement_curves({env_name: results["prob_improve_curves"]},
                                      {env_name: steps0},
                                      os.path.join(OUT, f"fig4b_prob_improve_curves_{env}.png"))
    print(f"\nRegenerated figures -> {OUT}")


if __name__ == "__main__":
    main()
