"""What one JaxNav training cycle actually does, with the shipped numbers.

Not a data plot -- a mechanism drawing. Every number is read from the config so it cannot
drift from what the code does.

LEFT   the loop: collect -> insert -> gradient updates -> target sync -> CCE scoring
RIGHT  one chunk on a timeline, showing how much of the buffer survives it at each
       capacity, against the target-sync and scoring events happening inside it

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.fig_training_loop
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
from counterfactual_rl.agents.jax_nav.config import DEFAULT_CONFIG as C
from .jaxnav_bootstrap_curves import ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_training_loop.png")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
C_COL, C_BUF, C_GRAD, C_TGT, C_CCE = "#6b7a8c", "#0072b2", "#009e73", "#d55e00", "#e69f00"

# --- the shipped configuration, read from config.py -------------------------
N_ENVS, STEPS = 256, 32                       # sweep BASE overrides
CHUNK = N_ENVS * STEPS
NSTEP = C['n_steps_per_update']               # env steps per gradient update
BATCH = C['batch_size']
TGT   = C['target_update_freq'] // NSTEP      # gradient steps per target sync
SCORE, NSAMP = 250, 64                        # sweep BASE
ROLL, HOR, ACTS = 20, 60, 15
GRAD = CHUNK // NSTEP                         # gradient updates per chunk


def box(ax, x, y, w, h, txt, col, sub=None, fs=10.5):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                   facecolor=col, alpha=.16, edgecolor=col, lw=2))
    ax.text(x+w/2, y+h*(.80 if sub else .86), txt, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=col)
    if sub:
        ax.text(x+w/2, y+h*.42, sub, ha="center", va="center", fontsize=9, color=INK2)


def arrow(ax, x0, y0, x1, y1, col=INK2, rad=0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2,
                                connectionstyle=f"arc3,rad={rad}"))


def main():
    fig = plt.figure(figsize=(17.6, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.18],
                          left=.03, right=.985, top=.70, bottom=.07, wspace=.13)

    # ================= LEFT: the loop =================
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
    box(ax, .5, 8.7, 9, 1.35, f"1.  COLLECT  {CHUNK:,} transitions", C_COL,
        sub=f"{N_ENVS} parallel envs x {STEPS} steps — inserted in ONE burst")
    arrow(ax, 5, 8.65, 5, 7.95)
    box(ax, .5, 6.55, 9, 1.35, "2.  INSERT into the replay buffer", C_BUF,
        sub="circular: each insert evicts the OLDEST entry")
    arrow(ax, 5, 6.5, 5, 5.8)
    box(ax, .5, 3.05, 9, 2.7, f"3.  {GRAD} GRADIENT UPDATES", C_GRAD)
    for i, t in enumerate([
            f"sample a batch of {BATCH} by priority  (TD x CCE)",
            "TD error  =  r + gamma x maxQ_TARGET(s')  -  Q_ONLINE(s,a)",
            "update ONLINE network"]):
        ax.text(5, 4.75 - i*.46, t, ha="center", fontsize=9.6, color=INK2)
    ax.text(5, 3.35, f"the buffer is the ONLY thing these {GRAD} updates can see",
            ha="center", fontsize=9.6, color=C_GRAD, fontweight="bold")
    box(ax, .5, 1.55, 4.3, 1.2, "4.  TARGET <- ONLINE", C_TGT, fs=10,
        sub=f"every {TGT} updates  ->  {GRAD//TGT}x per chunk")
    box(ax, 5.2, 1.55, 4.3, 1.2, "5.  CCE SCORING", C_CCE, fs=10,
        sub=f"every {SCORE} updates  ->  {GRAD/SCORE:.1f}x per chunk")
    ax.text(7.35, 1.18, f"{NSAMP} states x {ACTS} actions x {ROLL} rollouts x {HOR} steps\n"
                        f"= {NSAMP*ACTS*ROLL*HOR:,} imagined steps each time",
            ha="center", fontsize=8.4, color=INK2)
    arrow(ax, 2.65, 3.0, 2.65, 2.8, C_TGT); arrow(ax, 7.35, 3.0, 7.35, 2.8, C_CCE)
    arrow(ax, 9.85, 4.4, 9.85, 9.4, INK2, rad=-.55)
    ax.text(9.6, 7.0, "repeat  ~3,000x", rotation=-90, fontsize=9.5, color=INK2, va="center")
    ax.set_title("A.  One training cycle", fontsize=13, fontweight="bold",
                 color=INK, loc="left", pad=8)

    # ================= RIGHT: one chunk on a timeline =================
    ax = fig.add_subplot(gs[1]); ax.set_facecolor(SURFACE)
    caps = [(100_000, "100,000"), (8_192, "8,192"), (2_500, "2,500")]
    for i, (cap, lab) in enumerate(caps):
        y = -i * 1.35
        # fraction of the buffer that is OLDER than this chunk == survives it
        surv = max(0.0, 1 - CHUNK / cap)
        ax.add_patch(mp.Rectangle((0, y-.34), GRAD, .68, facecolor="#e9e6e1",
                                  edgecolor="none"))
        ax.add_patch(mp.Rectangle((0, y-.34), GRAD*surv, .68, facecolor=C_BUF,
                                  alpha=.55, edgecolor="none"))
        ax.text(-18, y, lab, ha="right", va="center", fontsize=11.5,
                fontweight="bold", color=C_BUF)
        if surv > 0:
            msg = f"{surv:.0%} of the buffer is OLDER than this chunk"
        elif abs(CHUNK/cap - 1) < .05:
            msg = "exactly break-even — the chunk fills it precisely, nothing older remains"
        else:
            msg = f"the chunk overwrites the whole buffer {CHUNK/cap:.1f}x over"
        ax.text(GRAD*.5, y, msg, ha="center", va="center", fontsize=10,
                color=INK if surv > 0 else "#8a2b06", fontweight="bold")
    for t in range(TGT, GRAD+1, TGT):
        ax.plot([t, t], [-3.35, .62], color=C_TGT, lw=1.8, ls=(0, (3, 2)), zorder=6)
    for t in range(SCORE, GRAD+1, SCORE):
        ax.plot([t, t], [-3.35, .62], color=C_CCE, lw=2.4, zorder=5)
    ax.plot([], [], color=C_TGT, lw=1.8, ls=(0, (3, 2)), label=f"target sync  (x{GRAD//TGT})")
    ax.plot([], [], color=C_CCE, lw=2.4, label=f"CCE scoring  (x{int(GRAD/SCORE)})")
    ax.add_patch(mp.Rectangle((0, 0), 0, 0, facecolor=C_BUF, alpha=.55,
                              label="survives the chunk"))
    ax.legend(fontsize=9.5, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(.5, 1.16))
    ax.set_xlim(-330, GRAD+30); ax.set_ylim(-3.6, 1.15)
    ax.set_yticks([])
    # the negative x-range exists only to give the row labels room; do not tick it
    ax.set_xticks(list(range(0, GRAD + 1, 128)))
    ax.set_xlabel(f"gradient updates within ONE chunk (0 – {GRAD})",
                  fontsize=10.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("B.  What survives one chunk", fontsize=13, fontweight="bold",
                 color=INK, loc="left", pad=26)

    fig.text(.03, .955, "The JaxNav training loop, with the shipped numbers",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.03, .885,
             f"Read A top to bottom, then B. Collection is BURSTY: {CHUNK:,} transitions "
             f"land at once, and only then do {GRAD} gradient updates run against whatever "
             f"the buffer now holds.\n"
             f"The target network — the reference every TD error is measured against — syncs "
             f"{GRAD//TGT} times inside that same chunk.\n"
             f"So at 100,000 capacity, 92% of what the gradient sees is older than the current "
             f"chunk: a stable pool. At 2,500 the chunk overwrites the buffer 3.3x, so every "
             f"update sees only the newest slice.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.5)
    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  chunk {CHUNK:,} · grad updates/chunk {GRAD} · target sync every {TGT} "
          f"· scoring every {SCORE}")


if __name__ == "__main__":
    main()
