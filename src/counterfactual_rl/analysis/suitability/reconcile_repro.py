"""Reconcile the suitability metrics with the validated Claim-1 result, using the REPRO
multi-seed checkpoints (paper/repro/cache/checkpoints/seed_{0,1,2}/{untrained,mid,trained}.pkl)
and the VALIDATED consequence setting: gamma=1.0 (win/lose grading), long horizon, n_rollouts=100.

Outputs:
  - reconciled_gain.png : CCE-vs-oracle scatter, 3 stages × 3 seeds (reproduces the Claim-1 figure)
  - prints a per-stage metric table (mean over seeds) with the FIXED gamma.
"""
import glob
import os

import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN
from counterfactual_rl.analysis.metrics import compute_consequence_metric
from counterfactual_rl.analysis.claim1.frozen_lake.oracle import compute_oracle
from counterfactual_rl.analysis.claim1.frozen_lake.score_states import _build_rollout_fn
from counterfactual_rl.analysis.suitability.rollout_sweep import (
    compute_abs_td_per_state, compute_visit_counts, greedy_actions)
from counterfactual_rl.analysis.suitability import metrics as M

REPRO = "paper/repro/cache/checkpoints"
SEEDS = [0, 1, 2]
STAGES = ["untrained", "mid", "trained"]
N_ROLL, HORIZON, GAMMA = 100, 500, 1.0   # VALIDATED consequence setting (win/lose grading)
SEED_COL = {0: "#2196F3", 1: "#ff9800", 2: "#e5534b"}


def returns_tensor(agent, states):
    fn = _build_rollout_fn(agent.env, agent.network, HORIZON, GAMMA)
    s = jnp.array(states, dtype=jnp.int32)
    a = jnp.arange(4, dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(0), len(states) * 4 * N_ROLL).reshape(len(states), 4, N_ROLL, 2)
    return np.array(fn(agent.params, s, a, keys))   # (S,4,N)


def cce_per_state(agent, returns, a_star):
    out = np.zeros(returns.shape[0])
    for i in range(returns.shape[0]):
        out[i] = compute_consequence_metric((int(a_star[i]),),
                    {(k,): returns[i, k] for k in range(4)},
                    metric="total_variation", aggregation="weighted_mean")
    return np.nan_to_num(out)


def main():
    _, oracle, nt = compute_oracle()                 # mean-gap oracle (Claim-1 def), slippery 8x8
    nt = list(nt)
    oracle_vals = np.array([oracle[s] for s in nt])

    data = {st: {} for st in STAGES}                 # data[stage][seed] = dict of arrays/metrics
    for seed in SEEDS:
        for st in STAGES:
            ck = os.path.join(REPRO, f"seed_{seed}", f"{st}.pkl")
            agent = FrozenLakeDQN(); agent.load(ck)
            states = np.array(nt, dtype=np.int32)
            R = returns_tensor(agent, states)
            a_star = greedy_actions(agent, states)
            C = M.stakes_C(R)
            cce = cce_per_state(agent, R, a_star)
            td = compute_abs_td_per_state(agent, states, a_star)
            d = compute_visit_counts(agent, n_episodes=40)[states]
            data[st][seed] = dict(
                cce=cce, C=C, td=td, d=d, R=R,
                gain=spearmanr(cce, oracle_vals).correlation,
                distinct=1 - abs(spearmanr(cce, td).correlation),
                gini=M.concentration(C)["gini"],
                snr=M.snr(R)["value"],
                need=spearmanr(C, d).correlation,
            )
            print(f"  seed{seed} {st:9s}: gain(CCE↔oracle)={data[st][seed]['gain']:.3f} "
                  f"distinct={data[st][seed]['distinct']:.2f} gini={data[st][seed]['gini']:.2f} "
                  f"snr={data[st][seed]['snr']:.1f} need={data[st][seed]['need']:.2f}", flush=True)

    # ---- figure: CCE vs oracle, 3 stage panels, colored by seed (reproduces Claim-1) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = {"untrained": "Untrained", "mid": "Mid-training", "trained": "Fully Trained"}
    for ax, st in zip(axes, STAGES):
        rhos = []
        for seed in SEEDS:
            dd = data[st][seed]
            ax.scatter(oracle_vals, dd["cce"], s=26, color=SEED_COL[seed], alpha=.8,
                       edgecolor="k", lw=.3, label=f"seed {seed}")
            rhos.append(dd["gain"])
        ax.set_xlabel("Oracle Q* consequence"); ax.set_ylabel("CCE score")
        ax.set_title(f"{labels[st]}   (mean ρ = {np.mean(rhos):.2f})", fontweight="bold")
        if st == "untrained": ax.legend(fontsize=8)
    fig.suptitle("RECONCILED — CCE vs Oracle with FIXED setting (γ=1.0, win/lose grading) · repro 3 seeds",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = "docs/figures/suitability/reconciled_gain.png"
    fig.savefig(out, dpi=130); print("wrote", out, flush=True)

    # ---- per-stage metric table (mean ± std over seeds) ----
    print("\n  stage      gain(CCE↔oracle)  distinct-TD   gini    snr     need   (mean±std over 3 seeds)")
    for st in STAGES:
        def ms(key):
            v = [data[st][s][key] for s in SEEDS]; return np.nanmean(v), np.nanstd(v)
        g=ms("gain"); di=ms("distinct"); gi=ms("gini"); sn=ms("snr"); ne=ms("need")
        print(f"  {st:9s}  {g[0]:.2f}±{g[1]:.2f}        {di[0]:.2f}±{di[1]:.2f}   "
              f"{gi[0]:.2f}±{gi[1]:.2f} {sn[0]:.0f}±{sn[1]:.0f}  {ne[0]:.2f}±{ne[1]:.2f}")
    print("SUITABILITY_RECONCILE_DONE", flush=True)


if __name__ == "__main__":
    main()
