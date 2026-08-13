"""CCE suitability predictor — CLI orchestration.

For each env (FL-det / FL-stoch) and each warmup checkpoint (random / 10% / 30% / full), load the
policy, roll out once to get returns(S,A,N), and compute the 6 v1 metrics. Emit scorecard.json +
a figure, and optionally inject real numbers into the interactive dashboard.

Usage:
    python -m counterfactual_rl.analysis.suitability.run_suitability \
        --run-id <id> [--runs-root .../frozen_lake/runs] \
        --envs FL-det FL-stoch --out scorecard.json --fig scorecard.png \
        [--dashboard-in docs/figures/mock_preview/dashboard.html --dashboard-out dashboard_real.html]
"""

import argparse
import glob
import os
import pickle
import re
from datetime import datetime, timezone

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.frozen_lake.consequence_dqn_vectorized import (
    FrozenLakeConsequenceDQNVectorized,
)
from counterfactual_rl.analysis.suitability.envs import make_frozenlake_adapter
from counterfactual_rl.analysis.suitability import metrics as M
from counterfactual_rl.analysis.suitability import scorecard as SC
from counterfactual_rl.analysis.suitability.rollout_sweep import (
    compute_return_tensor, greedy_actions, compute_cce_priority,
    compute_abs_td_per_state, compute_visit_counts, compute_horizon_sweep,
)

ENV_SLIPPERY = {"FL-det": False, "FL-stoch": True}
# Early-training fractional snapshots; the converged snapshot is the BEST-win checkpoint, NOT the
# last — FL DQN destabilizes late and the final checkpoint is often a collapsed 0%-win policy.
FRAC_PHASES = [("random", 0.0), ("10%", 0.1), ("30%", 0.3)]


def _parse_metrics_log(run_dir):
    """metrics.log → sorted [(episode, win_rate_percent)] (empty if absent)."""
    p = os.path.join(run_dir, "metrics.log")
    rows = []
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("#") or "episode" in line or not line.strip():
                continue
            parts = line.split()
            try:
                rows.append((int(parts[0]), float(parts[3].rstrip("%"))))
            except (ValueError, IndexError):
                pass
    return sorted(rows)


def _win_at(rows, ep):
    """Nearest-eval win rate (%) for a checkpoint episode, or None if no metrics.log."""
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r[0] - ep))[1]


def select_warmup_checkpoints(run_dir):
    """Return [(phase, episode, path, logged_win_rate)] — random/10%/30% by fraction plus a
    'best' phase = the highest-win-rate checkpoint (guards against late-training collapse)."""
    files = glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_*.pkl"))
    parsed = sorted((int(re.search(r"ckpt_(\d+)\.pkl", f).group(1)), f) for f in files)
    if not parsed:
        raise FileNotFoundError(f"no checkpoints under {run_dir}/checkpoints/")
    rows = _parse_metrics_log(run_dir)
    wins = [_win_at(rows, ep) for ep, _ in parsed]

    out, used = [], set()
    for name, frac in FRAC_PHASES:
        i = int(round(frac * (len(parsed) - 1)))
        ep, path = parsed[i]
        out.append((name, ep, path, wins[i]))
        used.add(path)
    # best = max nearest-eval win rate (fall back to last checkpoint if no metrics.log)
    if any(w is not None for w in wins):
        bi = int(np.nanargmax([w if w is not None else -1.0 for w in wins]))
    else:
        bi = len(parsed) - 1
    ep, path = parsed[bi]
    if path not in used:
        out.append(("best", ep, path, wins[bi]))
    else:  # best coincided with an early snapshot → still surface it explicitly
        out.append(("best", ep, path, wins[bi]))
    return out


def _load_config(path):
    with open(path, "rb") as f:
        return pickle.load(f)["config"]


def _load_params(agent, path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    agent.params = jax.tree.map(jnp.array, ck["params"])
    agent.target_params = jax.tree.map(jnp.array, ck["target_params"])


def build_agent_for_env(ref_ckpt, slippery, args):
    cfg = dict(_load_config(ref_ckpt))
    cfg["is_slippery"] = slippery
    if args.slip_prob is not None:
        cfg["slip_prob"] = args.slip_prob
    cfg["consequence_metric"] = args.metric
    if args.cf_n_rollouts:
        cfg["cf_n_rollouts"] = args.cf_n_rollouts
    agent = FrozenLakeConsequenceDQNVectorized(config=cfg)
    agent._build_rollout_fn()
    return agent


def compute_all_metrics(agent, adapter, args):
    states = adapter.states
    returns = compute_return_tensor(agent, states, batch=args.batch)
    a_star = greedy_actions(agent, states)
    C = M.stakes_C(returns)
    cce = compute_cce_priority(agent, returns, a_star)
    abs_td = compute_abs_td_per_state(agent, states, a_star)
    d = compute_visit_counts(agent, n_episodes=args.visit_episodes)
    d_at = d[states]

    if states.size <= args.horizon_states:
        sub = states
    else:
        sub = np.sort(np.random.default_rng(0).choice(states, args.horizon_states, replace=False))
    hsweep = compute_horizon_sweep(agent, sub, args.horizons)

    qss_at = adapter.qstar_spread[states] if adapter.qstar_spread is not None else None
    return {
        "concentration": M.concentration(C),
        "snr": M.snr(returns),
        "distinct_td": M.distinct_td(cce, abs_td),
        "gain_fidelity": M.gain_fidelity(cce, qss_at),
        "need": M.need(C, d_at),
        "horizon_fit": M.horizon_fit(agent.cf_horizon, hsweep),
    }


def run_env(run_dir, env_name, args):
    slippery = ENV_SLIPPERY[env_name]
    ckpts = select_warmup_checkpoints(run_dir)
    agent = build_agent_for_env(ckpts[0][2], slippery, args)
    adapter = make_frozenlake_adapter(agent, env_name, exact_truth=True)
    checkpoints = []
    for phase, ep, path, logged_win in ckpts:
        _load_params(agent, path)
        m = compute_all_metrics(agent, adapter, args)
        try:
            wr = float(agent.evaluate(n_episodes=args.eval_episodes)["win_rate"])  # 0..1, greedy
        except Exception:
            wr = None
        # Validation: the re-evaluated win rate (authoritative) vs metrics.log; flag mismatches
        # and degenerate 'best' policies (the past failure mode = using a collapsed checkpoint).
        warn = ""
        eval_pct = None if wr is None else 100.0 * wr
        if eval_pct is not None and logged_win is not None and abs(eval_pct - logged_win) > 30:
            warn += f"  ⚠ eval/log win MISMATCH ({eval_pct:.0f}% vs {logged_win:.0f}%)"
        if phase == "best" and eval_pct is not None and eval_pct < 20:
            warn += f"  ⚠ 'best' checkpoint is DEGENERATE (win={eval_pct:.0f}%) — run may have failed"
        checkpoints.append({"ckpt": os.path.basename(path), "episode": int(ep), "phase": phase,
                            "win_rate_eval": wr, "logged_win_pct": logged_win, "metrics": m})
        print(f"  [{env_name}] {phase:7s} ep={ep:6d}  win(eval)="
              f"{'?' if eval_pct is None else f'{eval_pct:.0f}%'} win(log)="
              f"{'?' if logged_win is None else f'{logged_win:.0f}%'}  ::  "
              f"conc={m['concentration']['gini']:.2f} snr={m['snr']['value']:.1f} "
              f"distinct={m['distinct_td']['value']} "
              f"gain={None if m['gain_fidelity'] is None else round(m['gain_fidelity']['value'],3)}"
              f"{warn}", flush=True)
    return {"slippery": slippery, "checkpoints": checkpoints}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="absolute run dir (overrides --run-id/--runs-root)")
    ap.add_argument("--run-id")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--envs", nargs="+", default=["FL-det", "FL-stoch"])
    ap.add_argument("--out", default="scorecard.json")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--dashboard-in", default=None)
    ap.add_argument("--dashboard-out", default=None)
    ap.add_argument("--metric", default="total_variation")
    ap.add_argument("--slip-prob", type=float, default=None)
    ap.add_argument("--cf-n-rollouts", type=int, default=None)
    ap.add_argument("--visit-episodes", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--horizon-states", type=int, default=12)
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        root = args.runs_root or os.path.join(
            os.path.dirname(__file__), "..", "..", "agents", "frozen_lake", "runs")
        run_dir = os.path.join(root, args.run_id)

    scorecard = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "run_dir": os.path.abspath(run_dir),
        "config": {"metric": args.metric, "slip_prob": args.slip_prob,
                   "cf_n_rollouts": args.cf_n_rollouts, "visit_episodes": args.visit_episodes,
                   "horizons": args.horizons},
        "envs": {},
    }
    for env_name in args.envs:
        print(f"== {env_name} ==", flush=True)
        scorecard["envs"][env_name] = run_env(run_dir, env_name, args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    SC.save_json(scorecard, args.out)
    print(f"wrote {args.out}", flush=True)
    if args.fig:
        SC.plot_scorecard(scorecard, args.fig)
        print(f"wrote {args.fig}", flush=True)
    if args.dashboard_in and args.dashboard_out:
        inj = SC.inject_dashboard(scorecard, args.dashboard_in, args.dashboard_out)
        print(f"injected {inj} into {args.dashboard_out}", flush=True)


if __name__ == "__main__":
    main()
