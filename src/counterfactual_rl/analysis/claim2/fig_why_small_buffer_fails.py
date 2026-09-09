"""Why shrinking the JaxNav buffer breaks learning — the two clocks.

A replay buffer is supposed to be the SLOW pool the network is fitted to, while the
target network is the anchor that moves slowly on top of it. Shrinking the buffer makes
the data churn as fast as the anchor moves, so there is nothing stable to converge to.

Panel 1  the two clocks, drawn over the same 1,250 gradient steps
Panel 2  how many SUCCESSFUL episodes are resident at once (JaxNav pays only at the goal,
         so these are the entire positive signal), measured at each arm's own peak
Panel 3  what actually happened

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.fig_why_small_buffer_fails
"""
import json, os, subprocess, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
from .jaxnav_bootstrap_curves import ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_why_small_buffer_fails.png")
R = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
MAN = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_capacity_outcome.json")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
COL = {'uniform': "#8a8a8a", 'cce_base': "#0072b2", 'cce_8192': "#e69f00", 'cce_2500': "#d55e00"}
CAP = {'uniform': 100_000, 'cce_base': 100_000, 'cce_8192': 8_192, 'cce_2500': 2_500}
ORDER = ['cce_base', 'cce_8192', 'cce_2500']
LAB = {'cce_base': "100,000", 'cce_8192': "8,192", 'cce_2500': "2,500"}
NSTEP, TARGET_STEPS, WINDOW = 16, 2000, 1250


def load():
    m = json.load(open(MAN)); aid = sorted({v['array_id'] for v in m.values()})[0]
    out = subprocess.run(["sacct", "-j", aid, "-X", "-n", "-P", "-o", "JobIDRaw,JobID,State"],
                         capture_output=True, text=True).stdout
    res = {p[1]: (p[0], p[2]) for p in (l.split("|") for l in out.strip().split("\n")) if len(p) >= 3}
    per = collections.defaultdict(list)
    for k, v in m.items():
        if v['job_id'] not in res: continue
        rd, s = res[v['job_id']]
        if s != 'COMPLETED': continue
        f = os.path.join(R, rd, 'metrics.log')
        if not os.path.exists(f): continue
        rows = []
        for line in open(f):
            if line.startswith('#') or not line.strip(): continue
            q = line.split()
            if len(q) >= 6 and q[0] != 'episode':
                rows.append((float(q[0]), float(q[3].rstrip('%')), float(q[4])))
        if len(rows) > 20: per[v['arm']].append(rows)
    return per


def main():
    per = load()
    tgt = TARGET_STEPS // NSTEP
    fig = plt.figure(figsize=(17.4, 6.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, .85, 1.15],
                          left=.045, right=.985, top=.60, bottom=.13, wspace=.30)

    # ---- 1. the two clocks ---------------------------------------------
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURFACE)
    for i, a in enumerate(ORDER):
        y = -i
        repl = CAP[a] / NSTEP                       # gradient steps to fully replace
        n_gen = max(1, int(np.ceil(WINDOW / repl)))
        for g in range(n_gen):                      # one block per buffer "generation"
            x0 = g * repl; w = min(repl, WINDOW - x0)
            if w <= 0: break
            ax.add_patch(mp.Rectangle((x0, y - .30), w, .60,
                                      facecolor=COL[a], alpha=.30 + .45*(g % 2),
                                      edgecolor=SURFACE, lw=1.6))
        ax.text(-40, y, LAB[a], ha="right", va="center", fontsize=11,
                fontweight="bold", color=COL[a])
        ax.text(WINDOW + 30, y, f"{n_gen if repl<WINDOW else 0} full swaps",
                va="center", fontsize=9.5, color=INK2)
    for t in range(0, WINDOW + 1, tgt):             # target-network refreshes
        ax.plot([t, t], [-2.55, .55], color=INK2, lw=1.1, ls=(0, (2, 2)), zorder=5)
    ax.text(WINDOW/2, .82, f"dashed = target network refresh  (every {tgt} steps)",
            ha="center", fontsize=10, color=INK2)
    ax.set_xlim(-380, WINDOW + 330); ax.set_ylim(-2.95, 1.15)
    ax.set_yticks([]); ax.set_xticks([0, 250, 500, 750, 1000, 1250])
    ax.set_xlabel("gradient steps", fontsize=10.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("1.  Each shade = one complete swap of the buffer",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)

    # ---- 2. successful episodes resident -------------------------------
    ax = fig.add_subplot(gs[1]); ax.set_facecolor(SURFACE)
    vals = []
    for a in ORDER:
        w, L = [], []
        for rows in per[a]:
            wr = np.array([r[1] for r in rows]); ln = np.array([r[2] for r in rows])
            i = int(np.convolve(wr, np.ones(21)/21, mode='valid').argmax()) + 10
            w.append(wr[i]); L.append(ln[i])
        vals.append(CAP[a] / np.median(L) * np.median(w) / 100)
    ax.barh(range(len(ORDER)), vals, .55, color=[COL[a] for a in ORDER])
    for i, v in enumerate(vals):
        ax.text(v * 1.25, i, f"{v:,.0f}", va="center", fontsize=12,
                fontweight="bold", color=COL[ORDER[i]])
    ax.set_xscale("log"); ax.set_xlim(4, 3000)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LAB[a] for a in ORDER], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("SUCCESSFUL episodes held at once  (log)", fontsize=10.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(axis="x", alpha=.55, color=GRID, lw=.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_title("2.  Examples of WINNING\n     available to the gradient",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)

    # ---- 3. the consequence --------------------------------------------
    ax = fig.add_subplot(gs[2]); ax.set_facecolor(SURFACE)
    grid = np.linspace(500, 250_000, 400)
    def smooth(y, w=21):
        # edge-PAD, not zero-pad: mode='same' pulls the last points toward zero and
        # drew a fake cliff at 250k on the 100,000 curve.
        y = np.asarray(y, float)
        if len(y) < w: return y
        return np.convolve(np.pad(y, w//2, mode="edge"), np.ones(w)/w,
                           mode="valid")[:len(y)]
    for a in ORDER:
        st = np.vstack([np.interp(grid, [r[0] for r in rows],
                                  smooth([r[1] for r in rows])) for rows in per[a]])
        ax.plot(grid, np.median(st, axis=0), lw=2.6, color=COL[a], label=LAB[a])
    ax.set_xlabel("episodes (thousands)", fontsize=10.5, color=INK2)
    ax.set_ylabel("win rate (%)", fontsize=10.5, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
    ax.set_ylim(0, 70); ax.legend(fontsize=10, frameon=False, title="buffer")
    ax.grid(alpha=.55, color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title("3.  What happened", fontsize=12.5, fontweight="bold",
                 color=INK, loc="left", pad=10)

    fig.text(.045, .955, "Why the small buffer fails: the ground moves as fast as the anchor",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.045, .885,
             "DQN fits the network to the buffer while a slow-moving TARGET NETWORK provides "
             "the reference. That only works if the buffer is the STABLE thing.\n"
             "Panel 1: at 100,000 the buffer never fully turns over between target refreshes; "
             "at 2,500 it is swapped ~8 times in the same span.\n"
             "Panel 2: JaxNav pays reward ONLY at the goal, so successful episodes are the "
             "entire positive signal — and a small buffer holds ~10 of them at a time, not ~690.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.5)
    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
