"""Does a stored CCE score go STALE, or just get old?  (gates the freshness arm)

A CCE score is written once and never recomputed; measured median staleness is ~1,788
Q-updates (~14 target-net refreshes). Whether that MATTERS depends on how fast a score
actually moves, which nothing had ever measured -- buffer snapshots hold only the
priority arrays, not the states, so it cannot be done after the fact.

cf_drift_probe_n freezes a FIXED set of 512 transitions (states copied out of the buffer)
and re-scores exactly those at every eval, logging the Q-values from the same forward
pass. Correlating score vectors at increasing separations gives rho(delta).

THE CONTROL IS THE POINT. If rho_CCE falls while rho_Q stays high, the network did not
move and the SCORE did -- that is estimator noise, not staleness, and re-scoring more
often would only re-roll the same dice.
"""
import glob
import json
import os

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
RUNS = os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(WT, "docs/figures/real/claim2/jaxnav")
# Our measured median staleness is 1,788 GRADIENT UPDATES. The probe logs episodes too,
# and its own log gives 6.48 updates per episode (median per-probe and overall agree), so
# that is ~276 EPISODES. Everything below is plotted in EPISODES -- the clock every other
# figure in this project uses -- and measured directly from the episode field rather than
# converted, so no ratio is baked in.
MEASURED_STALENESS_EP = 276


def load():
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*", "drift.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) < 20:
            continue
        out.append((os.path.basename(os.path.dirname(f)),
                    np.array([r["episode"] for r in rows]),
                    np.array([r["cce"] for r in rows]),
                    np.array([r["q_taken"] for r in rows])))
    return out


def rho_curve(upd, M, deltas, tol=.35):
    from scipy.stats import spearmanr
    med, lo, hi = [], [], []
    for d in deltas:
        vals = []
        for i in range(len(upd)):
            j = int(np.argmin(np.abs(upd - (upd[i] + d))))
            if j <= i or abs(upd[j] - upd[i] - d) > d * tol:
                continue
            if M[i].std() == 0 or M[j].std() == 0:
                continue
            vals.append(spearmanr(M[i], M[j]).statistic)
        if vals:
            med.append(np.mean(vals)); lo.append(np.percentile(vals, 25))
            hi.append(np.percentile(vals, 75))
        else:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    return np.array(med), np.array(lo), np.array(hi)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = load()
    if not runs:
        raise SystemExit("no drift.jsonl found")
    deltas = np.array([50, 100, 200, 276, 400, 800, 1500, 3000])
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
    C_CCE, C_Q = "#c0392b", "#1f6fb2"
    all_c, all_q = [], []
    for rid, upd, S, Q in runs:
        mc, *_ = rho_curve(upd, S, deltas)
        mq, *_ = rho_curve(upd, Q, deltas)
        all_c.append(mc); all_q.append(mq)
        ax.plot(deltas, mc, color=C_CCE, alpha=.28, lw=1.2)
        ax.plot(deltas, mq, color=C_Q, alpha=.28, lw=1.2)
    mc, mq = np.nanmean(all_c, axis=0), np.nanmean(all_q, axis=0)
    ax.plot(deltas, mc, color=C_CCE, lw=3, marker="o", ms=5,
            label=f"CCE score   (n={len(runs)} runs, 512 transitions each)")
    ax.plot(deltas, mq, color=C_Q, lw=3, marker="s", ms=5,
            label="Q-value of the taken action   (the CONTROL)")
    ax.axvline(MEASURED_STALENESS_EP, color="#444", ls="--", lw=1.2)
    i = int(np.argmin(np.abs(deltas - MEASURED_STALENESS_EP)))
    ax.annotate(f"our measured median staleness\n~276 episodes\n"
                f"CCE rho = {mc[i]:.2f}   ·   Q rho = {mq[i]:.2f}",
                xy=(MEASURED_STALENESS_EP, mc[i]), xytext=(430, .18),
                fontsize=9.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("Δ  —  EPISODES between the two scorings", fontsize=11)
    ax.set_ylabel("Spearman ρ  between the same 512 transitions", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.2, lw=.7, which="both")
    ax.legend(fontsize=9.5, loc="lower left")
    ax.set_title("Does a stored CCE score go stale?\n"
                 "same transitions, re-scored later — thin lines are individual runs",
                 fontsize=12, loc="left")
    fig.text(.012, .015,
             "The Q control is what makes this readable. At our real staleness the network has "
             f"barely moved (ρ_Q = {mq[i]:.2f}) while the score has kept less than half its "
             f"ordering (ρ_CCE = {mc[i]:.2f}).\n"
             "A score that reshuffles while the policy holds still is MONTE-CARLO NOISE in the "
             "estimator, not staleness — scoring more often would re-roll the same dice. "
             "The cheaper fix would be more rollouts per score.",
             fontsize=8.8, color="#55666f")
    fig.tight_layout(rect=[0, .085, 1, 1])
    p = os.path.join(OUT, "drift_probe.png")
    fig.savefig(p); print("wrote", p)
    print(f"  at delta={MEASURED_STALENESS_EP} episodes: rho_CCE={mc[i]:.3f}  rho_Q={mq[i]:.3f}")


if __name__ == "__main__":
    main()
