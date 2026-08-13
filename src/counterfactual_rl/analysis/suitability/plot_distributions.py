"""Plot the DISTRIBUTIONS behind each scorecard number (not just the summary).

For the BEST checkpoint of a det run and a stoch run, recompute the per-state raw arrays
(C, CCE-priority, |TD|, exact Q*-spread, visit freq, between/within variance) and plot the
actual scatter/histogram each metric is reduced from. Columns = FL-det vs FL-stoch.

Usage:
    python -m counterfactual_rl.analysis.suitability.plot_distributions \
        --det-run .../runs/257440 --stoch-run .../runs/255545 \
        --scorecard docs/figures/suitability/scorecard.json \
        --out docs/figures/suitability/distributions.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from counterfactual_rl.analysis.suitability.run_suitability import (
    build_agent_for_env, _load_params, select_warmup_checkpoints, ENV_SLIPPERY)
from counterfactual_rl.analysis.suitability.envs import make_frozenlake_adapter
from counterfactual_rl.analysis.suitability.rollout_sweep import (
    compute_return_tensor, greedy_actions, compute_cce_priority,
    compute_abs_td_per_state, compute_visit_counts)
from counterfactual_rl.analysis.suitability import metrics as M

DET_C, STO_C = "#2e7d32", "#e08a1e"


def raw_for(run_dir, env_name, args):
    best = [s for s in select_warmup_checkpoints(run_dir) if s[0] == "best"][0]
    phase, ep, path, win = best
    agent = build_agent_for_env(path, ENV_SLIPPERY[env_name], args)
    _load_params(agent, path)
    ad = make_frozenlake_adapter(agent, env_name, exact_truth=True)
    s = ad.states
    R = compute_return_tensor(agent, s, batch=args.batch)
    a = greedy_actions(agent, s)
    return dict(
        C=M.stakes_C(R), cce=compute_cce_priority(agent, R, a),
        td=compute_abs_td_per_state(agent, s, a),
        qss=ad.qstar_spread[s], d=compute_visit_counts(agent, n_episodes=args.visit_episodes)[s],
        between=R.mean(2).var(1), within=R.var(2).mean(1),
        ep=ep, win=win, n=len(s))


def gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    if n == 0 or x.sum() == 0: return 0.0
    c = np.cumsum(x); return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def rho(a, b):
    r = spearmanr(a, b).correlation
    return float("nan") if r is None else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-run", required=True)
    ap.add_argument("--stoch-run", required=True)
    ap.add_argument("--scorecard", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="total_variation")
    ap.add_argument("--slip-prob", type=float, default=None)
    ap.add_argument("--cf-n-rollouts", type=int, default=60)
    ap.add_argument("--visit-episodes", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    print("computing FL-det raw arrays...", flush=True)
    det = raw_for(args.det_run, "FL-det", args)
    print("computing FL-stoch raw arrays...", flush=True)
    sto = raw_for(args.stoch_run, "FL-stoch", args)

    means_by_h = {}
    if args.scorecard and os.path.exists(args.scorecard):
        sc = json.load(open(args.scorecard))
        for env in ("FL-det", "FL-stoch"):
            try:
                means_by_h[env] = sc["envs"][env]["checkpoints"][-1]["metrics"]["horizon_fit"]["means_by_h"]
            except Exception:
                means_by_h[env] = None

    cols = [("FL-det", det, DET_C), ("FL-stoch", sto, STO_C)]
    rows = ["Concentration", "SNR", "Distinct-TD", "Gain-fidelity", "NEED", "Horizon-fit"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 22))

    for j, (env, D, col) in enumerate(cols):
        axes[0, j].set_title(f"{env}   (best ckpt ep{D['ep']}, win {D['win']:.0f}%, {D['n']} states)",
                             fontsize=12, fontweight="bold", color=col)

        # 1 Concentration — histogram of per-state stakes C(s)
        ax = axes[0, j]
        ax.hist(D["C"], bins=20, color=col, alpha=.85)
        ax.set_xlabel("stakes C(s)"); ax.set_ylabel("# states")
        ax.text(.97, .9, f"Gini={gini(D['C']):.2f}", ha="right", transform=ax.transAxes, fontsize=11)

        # 2 SNR — per-state between vs within action variance (log-log)
        ax = axes[1, j]
        ax.scatter(D["within"] + 1e-6, D["between"] + 1e-6, s=22, color=col, alpha=.7, edgecolor="k", lw=.3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("within-action var (noise)"); ax.set_ylabel("between-action var (signal)")
        snr = D["between"].mean() / (D["within"].mean() + 1e-9)
        ax.text(.5, .04, f"SNR(agg)={min(snr,1e3):.2f}", ha="center", transform=ax.transAxes, fontsize=11)

        # 3 Distinct-TD — CCE vs |TD| (want NO trend)
        ax = axes[2, j]
        ax.scatter(D["td"], D["cce"], s=22, color=col, alpha=.7, edgecolor="k", lw=.3)
        ax.set_xlabel("|TD| error"); ax.set_ylabel("CCE priority")
        ax.text(.97, .9, f"1-|ρ|={1-abs(rho(D['cce'],D['td'])):.2f}", ha="right", transform=ax.transAxes, fontsize=11)

        # 4 Gain-fidelity — CCE vs exact Q*-spread (want a trend)
        ax = axes[3, j]
        ax.scatter(D["qss"], D["cce"], s=22, color=col, alpha=.7, edgecolor="k", lw=.3)
        ax.set_xlabel("exact Q*-spread (truth)"); ax.set_ylabel("CCE priority")
        ax.text(.97, .9, f"ρ={rho(D['cce'],D['qss']):.2f}", ha="right", transform=ax.transAxes, fontsize=11)

        # 5 NEED — stakes C(s) vs discounted visit freq d(s)
        ax = axes[4, j]
        ax.scatter(D["d"] + 1e-9, D["C"], s=22, color=col, alpha=.7, edgecolor="k", lw=.3)
        ax.set_xscale("log"); ax.set_xlabel("visit freq d(s) (log)"); ax.set_ylabel("stakes C(s)")
        ax.text(.97, .9, f"ρ={rho(D['C'],D['d']):.2f}", ha="right", transform=ax.transAxes, fontsize=11)

        # 6 Horizon-fit — mean stakes vs rollout horizon
        ax = axes[5, j]
        mh = means_by_h.get(env) if means_by_h else None
        if mh:
            hs = sorted(int(k) for k in mh)
            ax.plot(hs, [mh[str(h)] if str(h) in mh else mh[h] for h in hs], "o-", color=col)
            ax.set_xlabel("rollout horizon H"); ax.set_ylabel("mean_s C(s)")
        else:
            ax.text(.5, .5, "no horizon sweep in scorecard", ha="center", transform=ax.transAxes, color="gray")

    for i, name in enumerate(rows):
        axes[i, 0].annotate(name, xy=(-.34, .5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=13, fontweight="bold")
    fig.suptitle("Distributions BEHIND each scorecard number — FL-det vs FL-stoch (best checkpoints, real data)",
                 fontsize=14, fontweight="bold", y=.997)
    fig.tight_layout(rect=[.02, 0, 1, .995])
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
