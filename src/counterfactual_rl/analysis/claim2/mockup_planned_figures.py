"""MOCKUPS of the figures the instrumented run would produce. SYNTHETIC DATA.

Nothing here is measured. This exists so the layout and the decision each figure
drives can be reviewed BEFORE spending a GPU night collecting the real thing.
Every panel is watermarked; the numbers are invented placeholders.

Panels map to asks:
  A,B  coverage + staleness        this week's finding + the paper's staleness theory
  C,D  rollout vs eval outcomes    tests whether rollouts predict reality at all
  E    length by outcome           Jeremy 08/27
  F    realised replay draws       Jeremy 07/10 ("times each state is sampled")
  G    CCE vs TD contribution      Jeremy 08/20 ("normalize TD to 0-1")
  H    double-sampling             Jeremy 07/10, never built

Delete once the real figures exist.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/MOCKUP_planned_figures.png")

C_SCORED, C_INHERIT = "#0072b2", "#b9b3aa"
C_CCE, C_TD = "#0072b2", "#7d3c98"
C_GOAL, C_CRASH, C_TIMEOUT, C_TRUNC = "#2c6e52", "#b0521f", "#c9a227", "#8a94a1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"
HOT = "#b0521f"
rng = np.random.default_rng(7)


def stamp(ax):
    ax.text(.5, .5, "SYNTHETIC", transform=ax.transAxes, fontsize=21, color="#d9534f",
            alpha=.13, ha="center", va="center", rotation=26, fontweight="bold", zorder=99)


def tidy(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=.7, alpha=.6)
    ax.set_axisbelow(True)
    stamp(ax)


def decides(ax, text):
    ax.text(0, -.34, text, transform=ax.transAxes, fontsize=8, color=HOT,
            va="top", style="italic")


fig, axes = plt.subplots(2, 4, figsize=(22, 10.6))
fig.subplots_adjust(left=.042, right=.99, top=.795, bottom=.115, wspace=.27, hspace=.60)
ep = np.linspace(0, 250, 200)

# ---- A. buffer composition, ACROSS TRAINING (the #8 ask, done properly) ---
ax = axes[0][0]
# logged as a fixed-bin histogram every eval -> a 2D array (evals x bins).
# One snapshot cannot show whether the distribution improves or decays.
nb, ne = 44, 200
edges = np.linspace(0, .5, nb + 1)
ctr = (edges[:-1] + edges[1:]) / 2
H = np.zeros((nb, ne))
for k in range(ne):
    frac = k / ne
    spike = np.exp(-((ctr - .085) ** 2) / (2 * .004 ** 2)) * 9800
    tail = np.exp(-ctr / (.06 + .10 * frac)) * (40 + 190 * frac)
    H[:, k] = spike + tail
im = ax.imshow(np.log10(H + 1), aspect="auto", origin="lower", cmap="magma",
               extent=[0, 250, 0, .5])
cb = fig.colorbar(im, ax=ax, pad=.02, fraction=.046)
cb.set_label("log10 entries", fontsize=8, color=INK2)
cb.ax.tick_params(labelsize=7, colors=INK2)
ax.set_xlabel("episodes (thousands)", fontsize=9, color=INK2)
ax.set_ylabel("stored CCE score", fontsize=9, color=INK2)
ax.text(8, .43, "the bright line = the inherited mean,\nheld by ~98% of the buffer",
        fontsize=8, color="white")
tidy(ax, "A.  Buffer score distribution over training")
ax.grid(False)
decides(ax, "DECIDES: Jeremy's KDE ask (#8), logged EVERY eval so it is a trajectory,\n"
            "not a snapshot. Same data also draws a 5-checkpoint ridgeline, free.")

# ---- B. score vs staleness ------------------------------------------------
ax = axes[0][1]
stale = rng.uniform(0, 12, 700)
ax.scatter(stale, .18 + rng.normal(0, .05, 700), s=5, color=C_GOAL, alpha=.45,
           label="flat -> paper's staleness claim holds")
ax.scatter(stale, .30 - stale * .017 + rng.normal(0, .035, 700), s=5,
           color=HOT, alpha=.45, label="drifts -> stale scores are NOT close")
ax.set_xlabel("thousands of Q-updates since scored", fontsize=9, color=INK2)
ax.set_ylabel("stored CCE score", fontsize=9, color=INK2)
ax.set_ylim(0, .42)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
tidy(ax, "B.  Score vs STALENESS")
decides(ax, "DECIDES: tests the paper's own claim that stale scores stay close\n"
            "to current ones. Replaces the clock test, which random scoring killed.")

# ---- C. rollout outcomes --------------------------------------------------
ax = axes[0][2]
goal = .06 + .10 / (1 + np.exp(-(ep - 120) / 40))
crash = .70 - .16 / (1 + np.exp(-(ep - 110) / 45))
timeout = np.full_like(ep, .05)
ax.stackplot(ep, goal, crash, timeout, 1 - goal - crash - timeout,
             colors=[C_GOAL, C_CRASH, C_TIMEOUT, C_TRUNC],
             labels=["reached goal", "crashed", "200 cap", "horizon ran out"])
ax.set_xlim(0, 250); ax.set_ylim(0, 1)
ax.set_xlabel("episodes (thousands)", fontsize=9, color=INK2)
ax.set_ylabel("share of rollouts", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
tidy(ax, "C.  How the ROLLOUTS end")
decides(ax, "DECIDES: only green pays. If it never grows, no fix to the score\n"
            "helps and JaxNav is done.")

# ---- D. eval outcomes -----------------------------------------------------
ax = axes[0][3]
g = .05 + .55 / (1 + np.exp(-(ep - 95) / 28))
c = .55 - .28 / (1 + np.exp(-(ep - 100) / 30))
ax.stackplot(ep, g, c, 1 - g - c, colors=[C_GOAL, C_CRASH, C_TIMEOUT],
             labels=["goal (= win rate)", "crashed", "timed out"])
ax.set_xlim(0, 250); ax.set_ylim(0, 1)
ax.set_xlabel("episodes (thousands)", fontsize=9, color=INK2)
ax.set_ylabel("share of eval episodes", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
tidy(ax, "D.  How the EVAL episodes end")
decides(ax, "DECIDES: replaces my crash BOUND with the real number.\n"
            "Paired with C, tests whether rollouts predict reality AT ALL.")

# ---- E. length by outcome -------------------------------------------------
ax = axes[1][0]
ax.plot(ep, 175 - 85 / (1 + np.exp(-(ep - 90) / 30)), lw=2, color=C_TRUNC,
        ls=(0, (5, 3)), label="avg_length (all) — today's metric")
ax.plot(ep, 128 - 42 / (1 + np.exp(-(ep - 110) / 45)), lw=2.2, color=C_GOAL,
        label="avg_length_goal  ← Jeremy 08/27")
ax.plot(ep, 60 - 18 / (1 + np.exp(-(ep - 80) / 40)), lw=2, color=C_CRASH,
        label="avg_length_crash")
ax.axhline(200, color=HOT, lw=1, ls=(0, (4, 4)))
ax.set_xlim(0, 250); ax.set_ylim(0, 215)
ax.set_xlabel("episodes (thousands)", fontsize=9, color=INK2)
ax.set_ylabel("steps", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
tidy(ax, "E.  Length, conditional on outcome")
decides(ax, "DECIDES: separates 'navigates better' from 'stops dying early'.\n"
            "Dashed = today's confounded metric.")

# ---- F. realised replay draws ---------------------------------------------
ax = axes[1][1]
x = np.linspace(0, 1, 200)
ax.plot(x, x, lw=2, color=C_TRUNC, ls=(0, (5, 3)), label="uniform  (1.00)")
ax.plot(x, x ** 1.18, lw=2.2, color=C_SCORED, label="measured today  (0.92)")
ax.plot(x, x ** 2.6, lw=2, color=C_GOAL, label="real prioritising  (0.47)")
ax.fill_between(x, x ** 1.18, x, color=C_SCORED, alpha=.12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("buffer fraction, least-replayed first", fontsize=9, color=INK2)
ax.set_ylabel("cumulative share of draws", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
tidy(ax, "F.  What actually got replayed")
decides(ax, "DECIDES: priorities are what we INTEND; this is what the sampler DID.\n"
            "Jeremy 07/10. Needs the sampling.npz port.")

# ---- G. CCE vs TD spread ACROSS TRAINING  (Jeremy 08/20) -----------------
ax = axes[1][2]
cv_cce = .19 - .01 * np.exp(-(ep - 60) / 90)
cv_td = 2.5 + 17 / (1 + np.exp(-(ep - 70) / 45))
ax.plot(ep, cv_cce, lw=2.4, color=C_CCE, label="CCE score   std/mean")
ax.plot(ep, cv_td, lw=2.4, color=C_TD, label="TD error   std/mean")
ax.fill_between(ep, cv_cce, cv_td, color=C_TD, alpha=.08)
ax.set_yscale("log"); ax.set_ylim(.05, 60); ax.set_xlim(0, 250)
ax.set_xlabel("episodes (thousands)", fontsize=9, color=INK2)
ax.set_ylabel("relative spread (log)", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.annotate("", xy=(150, cv_cce[120]), xytext=(150, cv_td[120]),
            arrowprops=dict(arrowstyle="<->", color=HOT, lw=1.3))
ax.text(143, 2.0, "MEASURED\n52-129x", fontsize=8.5, color=HOT,
        fontweight="bold", ha="right")
tidy(ax, "G.  CCE vs TD spread, over training")
decides(ax, "DECIDES: Jeremy 08/20 — 'equal exponents do NOT imply equal\n"
            "contribution'. Rerun with TD normalised 0-1 and this gap should close.")

# ---- H. double-sampling  (Jeremy 07/10) -----------------------------------
ax = axes[1][3]
ax.plot(ep, .88 - .06 / (1 + np.exp(-(ep - 90) / 40)), lw=2.4, color=C_SCORED,
        label="CCE batch vs PER batch")
ax.axhline(.98, color=C_TRUNC, lw=1.6, ls=(0, (5, 3)))
ax.text(248, .952, "identical samplers", ha="right", fontsize=8, color=C_TRUNC)
ax.axhline(.35, color=C_GOAL, lw=1.6, ls=(0, (5, 3)))
ax.text(248, .375, "genuinely different samplers", ha="right", fontsize=8, color=C_GOAL)
ax.set_xlim(0, 250); ax.set_ylim(.2, 1.03)
ax.set_xlabel("episodes (thousands)", fontsize=9, color=INK2)
ax.set_ylabel("batch overlap", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
tidy(ax, "H.  Double-sampling: CCE vs PER")
decides(ax, "DECIDES: Jeremy 07/10, never built. Draw a PER batch alongside the\n"
            "CCE batch, DON'T apply it, compare. Direct — no ESS interpretation.")

fig.text(.042, .955, "MOCKUP  —  what the instrumented run would produce",
         fontsize=18, fontweight="bold", color=INK)
fig.text(.042, .919,
         "SYNTHETIC DATA THROUGHOUT. Nothing here is measured. Layout and decision logic only.",
         fontsize=11, color="#d9534f", fontweight="bold")
fig.text(.042, .886,
         "A: a 44-bin histogram logged EVERY eval (44 ints/line). B: ~6 lines.   C: ~8 lines in the rollout.   "
         "D,E: ~6 lines in evaluate() + a new eval.jsonl so uniform and PER get them too.",
         fontsize=9, color=INK2)
fig.text(.042, .862,
         "F: port log_sampling/snapshot_draws from research/cce-buffer-diagnosis (~30).   "
         "G: ~4 lines + a normalise-TD flag.   H: ~15 lines, open since 07/10.",
         fontsize=9, color=INK2)
fig.text(.042, .833,
         "G and H close asks Jeremy made on 08/20 and 07/10 that were never actioned. "
         "G's caption carries a number we already measured from the finished sweep.",
         fontsize=9, color=HOT, style="italic")

fig.savefig(OUT, dpi=150, facecolor="white")
print(f"wrote {OUT}")
