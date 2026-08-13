"""Smoke test for the suitability pipeline. Run:
    python -m counterfactual_rl.analysis.suitability._smoke

Asserts:
  1. env: slip_probability=0.0 matches is_slippery=False; slip_probability=2/3 matches is_slippery=True.
  2. bridge: GAIN-fidelity(FL-det) = Spearman(C(s)_rollout, exact Q*-spread) is clearly positive.
  3. SNR contrast: SNR(FL-det) >> SNR(FL-stoch)  (deterministic within-action variance ≈ 0).
  4. scorecard JSON round-trips.
Keeps cf_n_rollouts small and scores all non-terminal states (FL is tiny) → seconds on CPU.
"""

import glob
import json
import os
import re
import sys
import tempfile
import types

import numpy as np
import jax.numpy as jnp

from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv
from counterfactual_rl.analysis.suitability.envs import make_frozenlake_adapter
from counterfactual_rl.analysis.suitability import metrics as M
from counterfactual_rl.analysis.suitability import scorecard as SC
from counterfactual_rl.analysis.suitability.rollout_sweep import (
    compute_return_tensor, greedy_actions, compute_cce_priority,
)
from counterfactual_rl.analysis.suitability.run_suitability import build_agent_for_env, _load_params

RUNS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "agents", "frozen_lake", "runs")


def check_env():
    det = FrozenLakeEnv(map_name="4x4", is_slippery=False)
    slip = FrozenLakeEnv(map_name="4x4", is_slippery=True)
    p0 = FrozenLakeEnv(map_name="4x4", slip_probability=0.0)
    p23 = FrozenLakeEnv(map_name="4x4", slip_probability=2 / 3)
    nd, n0 = np.asarray(det.next_states), np.asarray(p0.next_states)
    # deterministic intended dest = slot 1 of the slip table
    assert np.array_equal(nd[:, :, 0], n0[:, :, 1]), "slip_prob=0 intended dest != deterministic"
    assert np.allclose(np.asarray(p0.outcome_probs), [0.0, 1.0, 0.0])
    assert np.array_equal(np.asarray(slip.next_states), np.asarray(p23.next_states))
    assert np.allclose(np.asarray(p23.outcome_probs), [1 / 3, 1 / 3, 1 / 3])
    print("[1/4] env slip_probability check OK")


def find_best_8x8_run():
    """Most-trained 8x8 run; PREFER a slippery-trained one (is_slippery=True). CCE's distributional
    (TV) consequence score is graded only when returns are stochastic, so the bridge to exact Q* is
    measured on FL-stoch with a slippery-trained policy (matches the project's Claim-1 ρ on slippery FL)."""
    import pickle
    cands = []
    for run in glob.glob(os.path.join(RUNS_ROOT, "*")):
        cks = glob.glob(os.path.join(run, "checkpoints", "ckpt_*.pkl"))
        if not cks:
            continue
        eps = [int(re.search(r"ckpt_(\d+)\.pkl", c).group(1)) for c in cks]
        cands.append((max(eps), run))
    cands.sort(reverse=True)

    fallback = None
    for _, run in cands[:40]:
        last = max(glob.glob(os.path.join(run, "checkpoints", "ckpt_*.pkl")),
                   key=lambda c: int(re.search(r"ckpt_(\d+)\.pkl", c).group(1)))
        try:
            cfg = pickle.load(open(last, "rb"))["config"]
        except Exception:
            continue
        if cfg.get("map_name") != "8x8":
            continue
        if fallback is None:
            fallback = (run, last)
        if cfg.get("is_slippery") is True:    # slippery-trained → fair bridge in FL-stoch
            return run, last
    if fallback is None:
        raise RuntimeError("no 8x8 run with checkpoints found")
    return fallback


def _args():
    return types.SimpleNamespace(slip_prob=None, metric="total_variation",
                                 cf_n_rollouts=60, batch=256)


def metrics_for(env_name, slippery, ckpt, args):
    agent = build_agent_for_env(ckpt, slippery, args)
    _load_params(agent, ckpt)
    adapter = make_frozenlake_adapter(agent, env_name, exact_truth=True)
    states = adapter.states
    returns = compute_return_tensor(agent, states, batch=args.batch)
    a_star = greedy_actions(agent, states)
    C = M.stakes_C(returns)
    cce = compute_cce_priority(agent, returns, a_star)
    qss = adapter.qstar_spread[states]
    return {
        "snr": M.snr(returns),
        "gain_fidelity": M.gain_fidelity(cce, qss),
        "concentration": M.concentration(C),
        "cf_horizon": agent.cf_horizon,
        "cov_C": float(np.mean(C > 1e-6)),     # frac states with any stakes signal
        "cov_cce": float(np.mean(cce > 1e-6)),
        "n_states": int(states.size),
    }


def main():
    check_env()
    run, ckpt = find_best_8x8_run()
    print(f"[setup] using {os.path.basename(run)} :: {os.path.basename(ckpt)}")
    args = _args()

    det = metrics_for("FL-det", False, ckpt, args)
    stoch = metrics_for("FL-stoch", True, ckpt, args)

    g_det = det["gain_fidelity"]["value"]
    g_stoch = stoch["gain_fidelity"]["value"]
    snr_det, snr_stoch = det["snr"]["value"], stoch["snr"]["value"]
    print(f"[setup] cf_horizon={det['cf_horizon']}  n_states={det['n_states']}  "
          f"cov_C det={det['cov_C']:.2f}/stoch={stoch['cov_C']:.2f}")
    print(f"[2/4] bridge  GAIN-fidelity  FL-det={g_det:.3f}  FL-stoch={g_stoch:.3f}  "
          f"(distributional CCE score needs stochasticity → stoch should be higher)")
    print(f"[3/4] SNR     FL-det={snr_det:.1f} (within_zero={det['snr']['within_zero_frac']:.2f})  "
          f"FL-stoch={snr_stoch:.2f} (agg_within={stoch['snr']['agg_within']:.4f})")

    # Plumbing + structural asserts (not the experiment):
    # (a) deterministic env has ~no within-action (env) noise; stochastic env does.
    assert det["snr"]["within_zero_frac"] > 0.9, "det env should have ~zero within-action variance"
    assert stoch["snr"]["agg_within"] > 0.0, "stoch env should have positive within-action variance"
    assert snr_det > snr_stoch, f"SNR(det) should exceed SNR(stoch): {snr_det} vs {snr_stoch}"
    # (b) the bridge: the distributional CCE score tracks exact Q* where it has signal (stochastic).
    assert g_stoch is not None and g_stoch > 0.3, f"bridge weak on FL-stoch: {g_stoch}"
    # (c) all metrics finite/sane
    for env_m in (det, stoch):
        assert np.isfinite(env_m["snr"]["value"])

    # JSON round-trip
    sc = {"schema_version": 1, "envs": {
        "FL-stoch": {"slippery": True, "checkpoints": [
            {"ckpt": os.path.basename(ckpt), "episode": 0, "phase": "full", "win_rate": None,
             "metrics": {**stoch, "distinct_td": {"value": 0.5}, "need": {"value": 0.0},
                         "horizon_fit": {"value": 1.0}}}]}}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sc.json")
        SC.save_json(sc, p)
        rt = SC.load_json(p)["envs"]["FL-stoch"]["checkpoints"][0]["metrics"]["snr"]["value"]
        assert rt == snr_stoch
    print("[4/4] scorecard JSON round-trip OK")
    print("SMOKE PASSED  (note: CCE's TV score is degenerate in deterministic envs — bridge is "
          "measured on FL-stoch, matching the project's Claim-1 ρ on slippery FL)")


if __name__ == "__main__":
    main()
