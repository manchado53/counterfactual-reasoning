"""CCE suitability — Connect Four driver (the player×foe grid).

For each FOE (random/rule_based/mcts) and each warmup checkpoint, build a FRESH agent (the foe is
baked into the rollout atom at build time), collect boards by self-play, score them, and compute
the cheap metrics. Emits a scorecard whose env keys are `C4-<foe>` so the existing
`scorecard.plot_scorecard` renders one line per foe, plus an opponent-sweep plot.

GAIN-fidelity = None (no oracle); horizon_fit = None (deferred — the atom bakes cf_horizon).

Stage usage (see plans/cce-suitability-connect-four.md):
  random/rule_based (cheap):
    python -m counterfactual_rl.analysis.suitability.run_suitability_c4 \
        --run-dir src/counterfactual_rl/agents/shared/runs/259285 --foes random rule_based \
        --out docs/figures/suitability/c4/scorecard_c4.json
  mcts (cut budget):  --foes mcts --cf-horizon 28 --mcts-n-sims 16 --n-rollouts 10
"""

import argparse
import os
from datetime import datetime, timezone

import numpy as np

from counterfactual_rl.analysis.suitability import metrics as M
from counterfactual_rl.analysis.suitability import scorecard as SC
from counterfactual_rl.analysis.suitability import c4_backbone as C4
from counterfactual_rl.analysis.suitability.run_suitability import select_warmup_checkpoints


def cell_metrics(agent, cfg, foe, args):
    gamma = float(cfg.get("gamma", 0.999))
    metric = cfg.get("consequence_metric", "total_variation")
    aggregation = cfg.get("consequence_aggregation", "weighted_mean")

    states, occ, phases, games = C4.collect_c4_states(
        agent, foe, args.n_boards, args.max_games, gamma, seed=args.seed, eps=args.collect_eps)
    returns = C4.compute_return_tensor_c4(agent, states, args.n_rollouts, args.chunk, args.seed + 1)
    C = M.stakes_C(returns)
    cce, greedy = C4.compute_cce_and_greedy(states, returns, metric, aggregation)
    abs_td = C4.compute_abs_td_c4(agent, states, greedy, foe, gamma, args.n_foe_replies, args.seed + 2)

    need = M.need(C, occ)
    need["mode"] = "sampled"
    m = {
        "concentration": M.concentration(C),
        "snr": M.snr(returns),
        "distinct_td": M.distinct_td(cce, abs_td),
        "gain_fidelity": None,                       # no oracle in C4
        "need": need,
        "horizon_fit": None,                         # deferred (atom bakes cf_horizon)
    }
    info = {"n_boards": len(states), "games": int(games),
            "median_C": round(float(np.nanmedian(C)), 4),
            "median_cce": round(float(np.nanmedian(cce)), 4)}
    return m, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--foes", nargs="+", default=["random", "rule_based", "mcts"])
    ap.add_argument("--best-only", action="store_true", help="single best checkpoint (timing probe)")
    ap.add_argument("--n-boards", type=int, default=600)
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--max-games", type=int, default=800)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--n-foe-replies", type=int, default=4)
    ap.add_argument("--collect-eps", type=float, default=0.1,
                    help="epsilon-greedy during board collection (diversifies vs deterministic foes)")
    ap.add_argument("--cf-horizon", type=int, default=None)      # mcts budget cut
    ap.add_argument("--mcts-n-sims", type=int, default=None)     # mcts budget cut
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/figures/suitability/c4/scorecard_c4.json")
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    ckpts = select_warmup_checkpoints(args.run_dir)
    if args.best_only:
        ckpts = [c for c in ckpts if c[0] == "best"][:1] or ckpts[-1:]
    cf_overrides = {}
    if args.cf_horizon is not None:
        cf_overrides["cf_horizon"] = args.cf_horizon
    if args.mcts_n_sims is not None:
        cf_overrides["mcts_n_sims"] = args.mcts_n_sims

    scorecard = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "run_dir": os.path.abspath(args.run_dir),
        "config": {"foes": args.foes, "n_boards": args.n_boards, "n_rollouts": args.n_rollouts,
                   "cf_overrides": cf_overrides, "n_foe_replies": args.n_foe_replies},
        "envs": {},
    }

    for foe in args.foes:
        env_key = f"C4-{foe}"
        print(f"== {env_key} ==", flush=True)
        checkpoints = []
        for phase, ep, path, logged_win in ckpts:
            agent, cfg = C4.load_agent(path, foe, cf_overrides)
            m, info = cell_metrics(agent, cfg, foe, args)
            checkpoints.append({"ckpt": os.path.basename(path), "episode": int(ep), "phase": phase,
                                "win_rate_eval": None, "logged_win_pct": logged_win, "metrics": m,
                                "info": info})
            sn = m["snr"]["value"]; dt = m["distinct_td"]["value"]
            print(f"  [{env_key}] {phase:7s} ep={ep:<5d} boards={info['n_boards']} "
                  f"medC={info['median_C']} snr={sn:.2f} "
                  f"distinct={'?' if dt is None else round(dt,3)} "
                  f"need={m['need']['value']}", flush=True)
        scorecard["envs"][env_key] = {"checkpoints": checkpoints}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    SC.save_json(scorecard, args.out)
    print(f"wrote {args.out}", flush=True)
    fig = args.fig or args.out.replace(".json", ".png")
    SC.plot_scorecard(scorecard, fig)
    print(f"wrote {fig}", flush=True)
    if hasattr(SC, "plot_opponent_sweep"):
        sweep = args.out.replace(".json", "_sweep.png")
        SC.plot_opponent_sweep(scorecard, sweep)
        print(f"wrote {sweep}", flush=True)


if __name__ == "__main__":
    main()
