"""Learning curves for the 6-cell JaxNav factorial, one panel per cell.

IQM across an arm's seeds on a common episode grid, so partially-trained arms
are visible for exactly as far as they have run. Arms still inside the epsilon
decay window (62.5k of 250k) are NOT comparable to finished ones -- the curve
shows that directly, which is the point of drawing it rather than tabulating
final numbers.

Reads run dirs via the manifest's `run_dir`, not `job_id`: SLURM array tasks are
allocated a distinct JobIDRaw and the trainer names its directory after that.
Run slurm/resolve_manifest.py first.
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
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_factorial_curves.png")

TARGET, DECAY = 250_000, 62_500
SIZES, FILLS = [(8, 8), (11, 11)], [0.1, 0.3, 0.5]
ARMS = ["uniform", "per", "cce_wmean", "cce_max", "cce_add", "cce_only"]
COL = {"uniform": "#6e6e6e", "per": "#0072b2", "cce_wmean": "#009e73",
       "cce_max": "#d55e00", "cce_add": "#bf8600", "cce_only": "#8452c4"}
LAB = {"uniform": "uniform", "per": "PER", "cce_wmean": "CCE mul·wmean",
       "cce_max": "CCE mul·max", "cce_add": "CCE add μ=.25", "cce_only": "CCE only μ=1"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def curves():
    man = json.load(open(MANIFEST))
    out = {}
    for rec in man.values():
        rd = rec.get("run_dir")
        if not rd:
            continue
        f = os.path.join(RUNS, str(rd), "metrics.log")
        if not os.path.exists(f):
            continue
        ep, wr = [], []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 5 or p[0] == "episode":
                continue
            ep.append(float(p[0])); wr.append(float(p[3].rstrip("%")) / 100)
        if len(ep) > 3:
            out.setdefault((rec["cell"], rec["arm"]), []).append(
                (np.array(ep), np.array(wr)))
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

    for r, size in enumerate(SIZES):
        for c, fill in enumerate(FILLS):
            cell = f"{size[0]}x{size[1]}_f{str(fill).replace('0.','0')}"
            ax = axes[r][c]
            any_data = False
            for arm in ARMS:
                runs = data.get((cell, arm), [])
                if not runs:
                    continue
                any_data = True
                # Extend as far as at least MIN_SEEDS have actually run, taking
                # the IQM over whichever seeds have reached each point. Using
                # min(endpoint) truncates a whole arm to its slowest seed --
                # which buried cce_only at 79k while 4 of its seeds were past
                # 90k. Seeds that have not reached a point simply do not vote.
                # clamp to how many seeds have actually started -- an arm with
                # 1 or 2 running seeds would otherwise index off the end
                MIN_SEEDS = min(len(runs), max(3, (len(runs) + 1) // 2))
                ends = sorted(e[-1] for e, _ in runs)
                hi = ends[len(ends) - MIN_SEEDS]
                grid = np.linspace(250, hi, 260)
                cols = []
                for g in grid:
                    vals = [np.interp(g, e, w) for e, w in runs if e[-1] >= g]
                    cols.append(iqm(np.array(vals)) if len(vals) >= 2 else np.nan)
                y = smooth(np.array(cols))
                done = sum(1 for e, _ in runs if e[-1] >= TARGET * .96)
                ax.plot(grid / 1000, y * 100, color=COL[arm], lw=2.2,
                        ls="-" if done == len(runs) else "--",
                        label=f"{LAB[arm]}  {done}/{len(runs)}")
            ax.axvspan(0, DECAY / 1000, color=GRID, alpha=.45, zorder=0)
            if any_data:
                ax.text(DECAY / 1000 / 2, 96, "exploring", ha="center", va="top",
                        fontsize=7.5, color=INK2, style="italic")
            ax.set_xlim(0, TARGET / 1000); ax.set_ylim(0, 100)
            ax.set_title(f"{size[0]}×{size[1]}   fill {fill}",
                         fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
            if any_data:
                ax.legend(frameon=False, fontsize=7.6, loc="upper left",
                          bbox_to_anchor=(0, .93), handlelength=1.5)
            else:
                ax.text(.5, .5, "not started", transform=ax.transAxes, ha="center",
                        va="center", fontsize=11, color="#b6b1a9")
            if r == 1:
                ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
            if c == 0:
                ax.set_ylabel("evaluation win rate (%)", fontsize=9.5, color=INK2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8.5)
            ax.grid(axis="y", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)

    fig.text(.055, .945, "JaxNav factorial — learning curves so far",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(.055, .900,
             "IQM across each arm's seeds. SOLID = all 5 seeds finished; DASHED = still "
             "running, so the curve stops where the shortest seed has reached.",
             fontsize=9.5, color=INK2)
    fig.text(.055, .866,
             "Shaded band is the epsilon-decay window (0-62.5k). An arm inside it has not "
             "finished exploring and cannot be compared to a finished one.",
             fontsize=9.5, color=INK2)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
