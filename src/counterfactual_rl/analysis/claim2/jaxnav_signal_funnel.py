"""How much of the replay buffer carries USABLE consequence signal, and what it steers.

Coverage (0.85%) is only the first bottleneck. Over half of what does get measured
comes back EXACTLY ZERO, so the share of the buffer carrying a non-zero consequence
score is about half the coverage figure again. Panel 1 is that funnel.

Panel 2 is the consequence: the measured entries receive LESS replay than their
population share (~0.83x). That is not a defect -- a transition measured at 0 SHOULD
be de-prioritised, and it is. It just means the measurement, on net, moves probability
away from the entries it was spent on, and the ~99% inherited-value bulk absorbs it.

Panel 3 places JaxNav against FrozenLake, the one environment where CCE wins. Two
points, no line: this is a comparison, not a trend, and n=2 environments cannot
support a claim about the relationship.

Everything comes from the buffer snapshots, which did not exist before 2026-09-03.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_signal_funnel
"""
import argparse
import glob
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import RUNS, ROOT

MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_instrument.json")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_signal_funnel.png")

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
C_BULK, C_MEAS, C_LIVE = "#c9c7c2", "#0072b2", "#009e73"
C_HOT = "#d55e00"
EPS, BETA = 0.01, 0.25
ARMS = ["cce_wmean", "cce_max", "cce_only", "cce_wmean_bs"]
SHORT = {"cce_wmean": "wmean", "cce_max": "max", "cce_only": "only μ=1",
         "cce_wmean_bs": "wmean +bs"}


def collect(array_id):
    man = json.load(open(MANIFEST))
    out = subprocess.run(["sacct", "-j", str(array_id), "-X", "-n", "-P",
                          "-o", "JobIDRaw,JobID,State"],
                         capture_output=True, text=True).stdout
    res = {p[1]: p[0] for p in (l.split("|") for l in out.strip().split("\n")) if len(p) >= 3}
    acc = {}
    for rec in man.values():
        if rec["arm"] not in ARMS or rec["job_id"] not in res:
            continue
        g = sorted(glob.glob(os.path.join(RUNS, res[rec["job_id"]], "buffer", "*.npz")))
        if not g:
            continue
        z = np.load(g[-1])
        cs, td, sa = z["consequence_scores"], z["td_magnitudes"], z["scored_at"]
        m = sa != -1
        pc = (cs + EPS) ** BETA; pc /= pc.sum()
        pt = (td + EPS) ** BETA; pt /= pt.sum()
        comb = pc if rec["arm"] == "cce_only" else pc * pt
        comb /= comb.sum()
        ep = int(os.path.basename(g[-1]).split("_")[1].split(".")[0])
        acc.setdefault(rec["arm"], []).append(dict(
            n=len(cs), ep=ep,
            frac_measured=float(m.mean()),
            frac_live=float((m & (cs > 0)).mean()),
            zero_of_measured=float((cs[m] <= 0).mean()),
            replay_share=float(comb[m].sum()),
        ))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274678")
    args = ap.parse_args()
    acc = collect(args.array)
    if not acc:
        print("no snapshots yet")
        return

    def med(arm, k):
        return float(np.median([r[k] for r in acc[arm]]))

    ref = ARMS[0] if ARMS[0] in acc else list(acc)[0]
    N = acc[ref][0]["n"]
    ep = int(np.median([r["ep"] for r in acc[ref]]))

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.2), facecolor=SURFACE)
    fig.subplots_adjust(left=.045, right=.985, top=.66, bottom=.115, wspace=.30)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    # ---- 1. the funnel --------------------------------------------------
    ax = axes[0]
    f_meas = med(ref, "frac_measured")
    f_live = med(ref, "frac_live")
    stages = [("every transition in the buffer", 1.0, C_BULK),
              ("has a MEASURED score", f_meas, C_MEAS),
              ("...and it is NON-ZERO", f_live, C_LIVE)]
    # LOG axis: 100% next to 0.28% on a linear scale renders the last two bars as
    # invisible slivers, which hides the very thing the panel is about.
    for i, (lab, v, col) in enumerate(stages):
        ax.barh(-i, 100 * v, .52, color=col, zorder=3)
        ax.text(100 * v * 1.25, -i, f"{v*N:,.0f}", va="center", fontsize=11,
                color=col, fontweight="bold", zorder=4)
        ax.text(0.00042, -i + .46, lab, fontsize=10.5, color=INK, fontweight="bold")
        ax.text(0.00042, -i - .40, f"{100*v:.2f}% of the buffer", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 3000)
    ax.set_ylim(-2.85, .95)
    ax.set_yticks([])
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%", "100%"], fontsize=9)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", alpha=.55, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("1.  Two bottlenecks, and they MULTIPLY", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left", pad=10)
    drop1 = f_meas
    drop2 = f_live / f_meas if f_meas else 0
    # Both annotations live to the RIGHT of every bar, where nothing else is drawn --
    # the earlier placement ran straight through the stage labels.
    ax.annotate("", xy=(100 * f_meas * 1.05, -0.90), xytext=(150, -0.18),
                arrowprops=dict(arrowstyle="->", color=C_HOT, lw=1.8,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(170, -0.42, f"x {drop1:.4f}\ncoverage", fontsize=10, color=C_HOT,
            fontweight="bold", va="center")
    ax.annotate("", xy=(100 * f_live * 1.05, -1.92), xytext=(150, -1.20),
                arrowprops=dict(arrowstyle="->", color=C_HOT, lw=1.8,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(170, -1.62, f"x {drop2:.2f}\n{100*(1-drop2):.0f}% of every\n"
                        f"measurement\nis exactly 0",
            fontsize=10, color=C_HOT, fontweight="bold", va="center")

    # ---- 2. replay share vs population share ----------------------------
    ax = axes[1]
    x = np.arange(len(ARMS))
    pop = [100 * med(a, "frac_measured") for a in ARMS if a in acc]
    rep = [100 * med(a, "replay_share") for a in ARMS if a in acc]
    names = [SHORT[a] for a in ARMS if a in acc]
    ax.bar(x - .19, pop, .36, color=C_BULK, label="share of the BUFFER they are")
    ax.bar(x + .19, rep, .36, color=C_MEAS, label="share of REPLAY they get")
    for i, (p, r) in enumerate(zip(pop, rep)):
        ax.text(i - .19, p + .012, f"{p:.2f}", ha="center", fontsize=9, color=INK2)
        ax.text(i + .19, r + .012, f"{r:.2f}", ha="center", fontsize=9,
                color=C_MEAS, fontweight="bold")
        ax.text(i, max(p, r) + .075, f"{r/p:.2f}x", ha="center", fontsize=10.5,
                color=C_HOT, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("% (of the buffer / of draws)", fontsize=10, color=INK2)
    ax.set_ylim(0, max(max(pop), max(rep)) * 1.35)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.55, color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("2.  The measured entries get LESS than their share",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)

    # ---- 3. JaxNav vs FrozenLake, two points, no line -------------------
    ax = axes[2]
    def belt(ns, si, st):
        lam = ns / (si * st)
        return 1 - (1 - np.exp(-lam)) / lam
    pts = [("JaxNav\nCCE = NULL", belt(64, 250, 16), med(ref, "zero_of_measured"), C_HOT),
           ("FrozenLake-det\nCCE = WINS", belt(128, 100, 4), 0.0, C_LIVE)]
    for lab, cov, zero, col in pts:
        ax.scatter([100 * cov], [100 * zero], s=290, color=col, zorder=5,
                   edgecolor=SURFACE, linewidth=2.5)
        ax.annotate(lab, (100 * cov, 100 * zero), textcoords="offset points",
                    xytext=(14, 10), fontsize=10.5, color=col, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("replay coverage: % of the buffer with a measured score",
                  fontsize=10, color=INK2)
    ax.set_ylabel("% of measurements that are exactly 0", fontsize=10, color=INK2)
    ax.set_xlim(.4, 40); ax.set_ylim(-7, 78)
    ax.grid(alpha=.55, color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.annotate("", xy=(11, 44), xytext=(1.15, 44),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4, ls=(0, (4, 2))))
    ax.text(3.2, 47, "raising coverage alone\nmoves you along here —\n"
                     "and buys more ZEROS",
            fontsize=9.5, color=INK2, ha="center")
    ax.set_title("3.  TWO POINTS, no line", fontsize=12.5, fontweight="bold",
                 color=INK, loc="left", pad=10)

    fig.text(.045, .955, "How much of the buffer carries usable CCE signal",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.045, .900,
             f"JaxNav 8x8 fill 0.3  ·  array {args.array}  ·  from the raw buffer "
             f"snapshots  ·  median of 3 seeds  ·  at ~{ep:,} of 250,000 episodes.\n\n"
             "Coverage is the first bottleneck and the zero-rate is the second, and "
             "they multiply. Panel 2 is not a defect: a transition measured at 0 SHOULD\n"
             "be de-prioritised, and it is — but that means the measurement moves "
             "probability AWAY from the entries it was spent on, into the inherited bulk.\n"
             "Panel 3 is two environments. It cannot support a claim about the "
             "relationship, only the observation that the two differ on both axes at once.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.5)

    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
