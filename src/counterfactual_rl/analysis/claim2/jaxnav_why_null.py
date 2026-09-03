"""Why the bootstrap fix changed the score but not the learning.

Three panels, left to right, following the causal chain:

  1  what ess_frac means as an EFFECTIVE BUFFER SIZE. Prioritising is supposed
     to shrink the pool you draw from. Ours barely shrinks it.
  2  the three quantities, before and after bootstrap. Only the first moved.
  3  the realised sampling concentration across all 250k episodes -- flat.

Every number is measured. The reference marks in panel 1 come from this
project's own runs, not from illustration: 0.60 is the target the ESS-matched
balance sweep pinned every arm to (lab-notebook 2026-08-19), and 0.47 is what
pure-CCE produced at a common exponent in that same entry.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_why_null
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import ROOT
from .jaxnav_bootstrap_ess import load as load_ess

OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_why_null.png")

CAP = 100_000
C_OFF, C_ON = "#0072b2", "#d55e00"
INK, INK2, GRID, SUNK = "#0b0b0b", "#52514e", "#dcdad5", "#f1efec"
HOT, COOL = "#b0521f", "#6b7a8c"

# measured, cce_wmean @ 8x8_f03, array 274476 (median of 5 seeds)
ESS_OFF, ESS_ON = 0.9153, 0.9222
MEAN_OFF, MEAN_ON = 0.0751, 0.2154
WIN_OFF, WIN_ON = 55.4, 55.6
PVAL = 0.837


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    fig.subplots_adjust(left=.058, right=.985, top=.70, bottom=.115, wspace=.26)

    # ---- 1. effective buffer size ---------------------------------------
    ax = axes[0]
    marks = [(1.00, "no prioritising at all", COOL),
             (ESS_OFF, "MEASURED  (both arms)", C_OFF),
             (0.60, "ESS-matched sweep target", INK2),
             (0.47, "pure CCE at a common exponent", INK2)]
    for i, (v, lab, col) in enumerate(marks):
        hot = "MEASURED" in lab
        ax.barh(i + .14, v * CAP, height=.40, color=col, alpha=.95 if hot else .38)
        ax.text(v * CAP + 1800, i + .14, f"{v*CAP:,.0f}",
                va="center", fontsize=10.5, color=col,
                fontweight="bold" if hot else "normal")
        ax.text(0, i - .22, lab, va="center", ha="left", fontsize=9.5,
                color=INK if hot else INK2,
                fontweight="bold" if hot else "normal")
    ax.set_ylim(-.62, len(marks) - .20); ax.invert_yaxis()
    ax.set_xlim(0, CAP * 1.20); ax.set_yticks([])
    ax.set_xticks([0, 25_000, 50_000, 75_000, 100_000])
    ax.set_xticklabels(["0", "25k", "50k", "75k", "100k"])
    ax.set_xlabel("effective number of transitions being drawn from",
                  fontsize=10, color=INK2)
    ax.set_title("1.  Prioritising should SHRINK the pool", fontsize=12,
                 fontweight="bold", color=INK, loc="left", pad=10)
    ax.annotate("", xy=(ESS_OFF * CAP, .62), xytext=(CAP, .62),
                arrowprops=dict(arrowstyle="<->", color=HOT, lw=1.4))
    ax.text(CAP * .955, .82, "we shrank it by 8%", ha="right", fontsize=9.5,
            color=HOT, fontweight="bold")

    # ---- 2. only one thing moved ----------------------------------------
    ax = axes[1]
    rows = [("mean CCE score", MEAN_OFF, MEAN_ON, "{:.3f}"),
            ("ess_frac\n(what gets replayed)", ESS_OFF, ESS_ON, "{:.3f}"),
            ("final win rate %", WIN_OFF, WIN_ON, "{:.1f}")]
    for i, (lab, a, b, fmt) in enumerate(rows):
        rel = b / a                       # each row normalised to its OFF value
        ax.barh(i - .17, 1.0, height=.30, color=C_OFF, alpha=.85)
        ax.barh(i + .17, rel, height=.30, color=C_ON, alpha=.85)
        ax.text(1.02, i - .17, fmt.format(a), va="center", fontsize=9.5, color=C_OFF)
        ax.text(rel + .04, i + .17, fmt.format(b), va="center", fontsize=9.5, color=C_ON)
        ax.text(-.06, i, lab, va="center", ha="right", fontsize=10, color=INK)
        chg = (rel - 1) * 100
        ax.text(3.34, i, f"{chg:+.0f}%", va="center", ha="right", fontsize=13,
                fontweight="bold", color=HOT if abs(chg) > 20 else INK2)
    ax.axvline(1.0, color=GRID, lw=1)
    ax.set_xlim(0, 3.45); ax.set_ylim(-.6, 2.6); ax.invert_yaxis()
    ax.set_yticks([]); ax.set_xticks([1.0, 2.0, 3.0])
    ax.set_xticklabels(["same", "2x", "3x"])
    ax.set_xlabel("relative to bootstrap OFF", fontsize=10, color=INK2)
    ax.set_title("2.  Only the score moved", fontsize=12, fontweight="bold",
                 color=INK, loc="left", pad=10)
    ax.text(3.34, -.48, "change", ha="right", fontsize=9, color=INK2)
    ax.text(.03, 2.44, f"paired over 20 seeds:  p = {PVAL}", fontsize=9.5,
            color=INK2, style="italic")

    # ---- 3. it never moved, all run -------------------------------------
    ax = axes[2]
    data = load_ess("274476")
    for arm, col, lab in (("cce_wmean", C_OFF, "bootstrap OFF"),
                          ("cce_wmean_bs", C_ON, "bootstrap ON")):
        runs = data.get(arm, [])
        if not runs:
            continue
        g = np.linspace(500, 250_000, 260)
        stack = np.array([np.interp(g, e, c["ess_frac"]) for e, c in runs])
        ax.plot(g / 1000, np.median(stack, 0), lw=2.2, color=col, label=lab)
    ax.axhline(1.0, color=COOL, lw=1.6, ls=(0, (5, 4)))
    ax.text(248, 1.007, "uniform sampling", ha="right", fontsize=9.5, color=COOL)
    ax.axhspan(0.9, 1.02, color=SUNK, zorder=0)
    ax.axvspan(0, 62.5, color=GRID, alpha=.45, zorder=0)
    ax.text(31, .445, "exploring", ha="center", fontsize=9, color=INK2, style="italic")
    ax.set_xlim(0, 250); ax.set_ylim(.42, 1.03)
    ax.set_xlabel("episodes (thousands)", fontsize=10, color=INK2)
    ax.set_ylabel("ess_frac", fontsize=10, color=INK2)
    ax.set_title("3.  And it never moved, for 250k episodes", fontsize=12,
                 fontweight="bold", color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.annotate("both arms sit here,\non top of each other",
                xy=(150, .918), xytext=(120, .62), fontsize=9.5, color=HOT,
                arrowprops=dict(arrowstyle="->", color=HOT, lw=1.3))

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_axisbelow(True)

    fig.text(.052, .935, "Why a real fix to the score changed nothing",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(.052, .885,
             "JaxNav 8×8 fill 0.3, CCE mul·wmean, 5 seeds, array 274476. "
             "The score decides WHICH transitions get replayed — so a better score only "
             "helps if it changes what the network sees.",
             fontsize=10.5, color=INK2)
    fig.text(.052, .845,
             "Bootstrap tripled the score (panel 2) but left the replay distribution "
             "untouched (panels 1 and 3), so training saw the same data either way.",
             fontsize=10.5, color=INK2)
    fig.text(.052, .795,
             "Reference marks in panel 1 are this project's own measurements: 0.60 is the "
             "target the ESS-matched balance sweep pinned every arm to, 0.47 is pure CCE at a "
             "common exponent (lab-notebook 2026-08-19).",
             fontsize=9.5, color=INK2, style="italic")

    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
