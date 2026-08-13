"""Run the suitability pipeline MATCHED: a det-trained policy in the det env (FL-det) and a
slippery-trained policy in the slippery env (FL-stoch), then merge into one scorecard + figure +
dashboard injection. (run_suitability probes one policy across envs; this probes each env with a
policy trained for it — the fair comparison.)

Usage:
    python -m counterfactual_rl.analysis.suitability.run_full \
        --det-run <det_run_dir> --stoch-run <stoch_run_dir> \
        --out scorecard.json --fig scorecard.png \
        [--dashboard-in dashboard.html --dashboard-out dashboard_real.html]
"""

import argparse
import os
from datetime import datetime, timezone

from counterfactual_rl.analysis.suitability import scorecard as SC
from counterfactual_rl.analysis.suitability.run_suitability import run_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-run", required=True, help="run dir for the FL-det (is_slippery=False) policy")
    ap.add_argument("--stoch-run", required=True, help="run dir for the FL-stoch (slippery) policy")
    ap.add_argument("--out", default="scorecard.json")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--dashboard-in", default=None)
    ap.add_argument("--dashboard-out", default=None)
    ap.add_argument("--metric", default="total_variation")
    ap.add_argument("--slip-prob", type=float, default=None)
    ap.add_argument("--cf-n-rollouts", type=int, default=60)
    ap.add_argument("--visit-episodes", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--horizon-states", type=int, default=16)
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    sc = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "config": {"metric": args.metric, "cf_n_rollouts": args.cf_n_rollouts,
                   "visit_episodes": args.visit_episodes, "horizons": args.horizons,
                   "det_run": os.path.abspath(args.det_run),
                   "stoch_run": os.path.abspath(args.stoch_run)},
        "envs": {},
    }
    print("== FL-det (det-trained policy, det env) ==", flush=True)
    sc["envs"]["FL-det"] = run_env(args.det_run, "FL-det", args)
    print("== FL-stoch (slippery-trained policy, slippery env) ==", flush=True)
    sc["envs"]["FL-stoch"] = run_env(args.stoch_run, "FL-stoch", args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    SC.save_json(sc, args.out)
    print(f"wrote {args.out}", flush=True)
    if args.fig:
        SC.plot_scorecard(sc, args.fig)
        print(f"wrote {args.fig}", flush=True)
    if args.dashboard_in and args.dashboard_out:
        inj = SC.inject_dashboard(sc, args.dashboard_in, args.dashboard_out)
        print(f"injected {inj} → {args.dashboard_out}", flush=True)


if __name__ == "__main__":
    main()
