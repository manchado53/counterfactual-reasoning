"""Mean EVAL EPISODE LENGTH for the 6-cell JaxNav factorial, one panel per cell.

Companion to `jaxnav_factorial_curves.py`, same grid and same colours, so the two
figures stack. Win rate alone cannot say HOW a policy fails; length can, because
an episode ends on `collisions | goal_reached | time_up` (jaxmarl jaxnav_env):

    short  + low win rate  ->  the robot is crashing
    long   + low win rate  ->  the robot is wandering to the 200-step cap
    short  + high win rate ->  the robot is going straight to the goal

Column 5 of metrics.log, already written by every run -- no re-evaluation needed.
(Column 6, avg_return, is NOT plotted: under sparse_reward with goal_rew=1.0 and
coll_rew=0.0 it equals win_rate exactly -- verified max |diff| = 0 over all 180
runs -- so it is a duplicate of the win-rate figure.)

Reads run dirs via the manifest's `run_dir`, not `job_id`: SLURM array tasks are
allocated a distinct JobIDRaw and the trainer names its directory after that.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compute_metrics import iqm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_factorial.json")
RUNS = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_factorial_length.png")

TARGET, DECAY, MAX_STEPS = 250_000, 62_500, 200
SIZES, FILLS = [(8, 8), (11, 11)], [0.1, 0.3, 0.5]
ARMS = ["uniform", "per", "cce_wmean", "cce_max", "cce_add", "cce_only"]
COL = {"uniform": "#6e6e6e", "per": "#0072b2", "cce_wmean": "#009e73",
       "cce_max": "#d55e00", "cce_add": "#bf8600", "cce_only": "#8452c4"}
LAB = {"uniform": "uniform", "per": "PER", "cce_wmean": "CCE mul·wmean",
       "cce_max": "CCE mul·max", "cce_add": "CCE add μ=.25", "cce_only": "CCE only μ=1"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def curves():
    """(cell, arm) -> [(episodes, avg_length), ...] one entry per seed."""
    man = json.load(open(MANIFEST))
    out = {}
    for rec in man.values():
        rd = rec.get("run_dir")
        if not rd:
            continue
        f = os.path.join(RUNS, str(rd), "metrics.log")
        if not os.path.exists(f):
            continue
        ep, ln = [], []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 6 or p[0] == "episode":
                continue
            ep.append(float(p[0])); ln.append(float(p[4]))
        if len(ep) > 3:
            out.setdefault((rec["cell"], rec["arm"]), []).append(
                (np.array(ep), np.array(ln)))
    return out


def smooth(y, w=21):
    if len(y) < w:
        return y
    pad = w // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(w) / w, mode="valid")[:len(y)]


def main():
    data = curves()
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.0))
    fig.subplots_adjust(left=.055, right=.99, top=.80, bottom=.075, hspace=.34, wspace=.13)

    grid = np.linspace(250, TARGET, 260)
    for r, size in enumerate(SIZES):
        for c, fill in enumerate(FILLS):
            cell = f"{size[0]}x{size[1]}_f{str(fill).replace('0.','0')}"
            ax = axes[r][c]
            for arm in ARMS:
                runs = data.get((cell, arm), [])
                if not runs:
                    continue
                # thin per-seed traces: the IQM hides the split between a seed
                # that crashes fast and one that wanders to the cap, and that
                # split is the reason for drawing this figure at all.
                for e, l in runs:
                    ax.plot(grid / 1000, smooth(np.interp(grid, e, l)),
                            color=COL[arm], lw=.7, alpha=.20, zorder=1)
                cols = [iqm(np.array([np.interp(g, e, l) for e, l in runs]))
                        for g in grid]
                ax.plot(grid / 1000, smooth(np.array(cols)), color=COL[arm],
                        lw=2.2, label=f"{LAB[arm]}  {len(runs)}/5", zorder=3)
            ax.axhline(MAX_STEPS, color="#b02020", lw=1.0, ls=(0, (5, 4)), zorder=2)
            ax.text(TARGET / 1000 - 3, MAX_STEPS - 4, "timeout cap (200)", ha="right",
                    va="top", fontsize=7.5, color="#b02020")
            ax.axvspan(0, DECAY / 1000, color=GRID, alpha=.45, zorder=0)
            ax.text(DECAY / 1000 / 2, 203, "exploring", ha="center", va="top",
                    fontsize=7.5, color=INK2, style="italic")
            ax.set_xlim(0, TARGET / 1000); ax.set_ylim(40, 210)
            ax.set_title(f"{size[0]}×{size[1]}   fill {fill}",
                         fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
            ax.legend(frameon=False, fontsize=7.6, loc="lower left",
                      bbox_to_anchor=(0, .01), handlelength=1.5, ncol=2,
                      columnspacing=1.0)
            if r == 1:
                ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
            if c == 0:
                ax.set_ylabel("mean eval episode length (steps)", fontsize=9.5, color=INK2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8.5)
            ax.grid(axis="y", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)

    fig.text(.055, .945, "JaxNav factorial — how long an evaluation episode lasts",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(.055, .900,
             "Bold = IQM over the arm's 5 seeds; thin = the individual seeds. Same 180 runs, "
             "same colours and grid as the win-rate figure.",
             fontsize=9.5, color=INK2)
    fig.text(.055, .866,
             "An episode ends on crash, goal, or the 200-step cap. Falling length = reaching "
             "the goal faster. Length pinned near 200 with a low win rate = wandering, not crashing.",
             fontsize=9.5, color=INK2)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
