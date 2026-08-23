"""Final win rate per arm, per cell -- the factorial's results table as a figure.

One panel per cell, one row per arm: individual seeds as dots, median as a bar.
Uniform is drawn as a reference line across every panel because it is the
baseline that actually matters here -- prioritised replay has to beat RANDOM
replay to be worth anything, and on the cluttered 8x8 cells it does not.

Win rate is the mean of each run's LAST 20 evaluations, not its last one. A
single eval is 100 episodes on a policy that oscillates; using it moved arm
medians by up to 24 points between adjacent evals.

    python -m counterfactual_rl.analysis.claim2.jaxnav_factorial_summary
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_factorial.json")
RUNS = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_factorial_summary.png")

TARGET, WINDOW = 250_000, 20
CELLS = ["8x8_f01", "8x8_f03", "8x8_f05", "11x11_f01", "11x11_f03", "11x11_f05"]
TITLE = {"8x8_f01": "8×8   fill 0.1", "8x8_f03": "8×8   fill 0.3", "8x8_f05": "8×8   fill 0.5",
         "11x11_f01": "11×11   fill 0.1", "11x11_f03": "11×11   fill 0.3",
         "11x11_f05": "11×11   fill 0.5"}
ARMS = ["uniform", "per", "cce_wmean", "cce_max", "cce_add", "cce_only"]
LAB = {"uniform": "uniform", "per": "PER", "cce_wmean": "CCE mul·wmean",
       "cce_max": "CCE mul·max", "cce_add": "CCE add μ=.25", "cce_only": "CCE only μ=1"}
COL = {"uniform": "#6e6e6e", "per": "#0072b2", "cce_wmean": "#009e73",
       "cce_max": "#d55e00", "cce_add": "#bf8600", "cce_only": "#8452c4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def results():
    man = json.load(open(MANIFEST))
    out = {}
    for rec in man.values():
        rd = rec.get("run_dir")
        if not rd:
            continue
        f = os.path.join(RUNS, str(rd), "metrics.log")
        if not os.path.exists(f):
            continue
        rows = []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 5 or p[0] == "episode":
                continue
            rows.append((float(p[0]), float(p[3].rstrip("%"))))
        if not rows or rows[-1][0] < TARGET * 0.96:      # finished runs only
            continue
        out.setdefault((rec["cell"], rec["arm"]), []).append(
            float(np.mean([w for _, w in rows[-WINDOW:]])))
    return out


def main():
    res = results()
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
    fig.subplots_adjust(left=.115, right=.985, top=.80, bottom=.07, hspace=.36, wspace=.22)
    rng = np.random.default_rng(0)

    for k, cell in enumerate(CELLS):
        ax = axes[k // 3][k % 3]
        uni = res.get((cell, "uniform"))
        if uni:
            ax.axvline(np.median(uni), color=COL["uniform"], ls="--", lw=1.4, zorder=1)
        for i, arm in enumerate(ARMS):
            y = len(ARMS) - 1 - i
            v = res.get((cell, arm))
            if not v:
                ax.text(2, y, "not finished", va="center", fontsize=8, color="#b6b1a9",
                        style="italic")
                continue
            med = np.median(v)
            ax.barh(y, med, height=.55, color=COL[arm], alpha=.28, zorder=2)
            ax.scatter(v, y + (rng.random(len(v)) - .5) * .30, s=22, color=COL[arm],
                       zorder=4, edgecolor="white", linewidth=.7)
            ax.plot([med, med], [y - .30, y + .30], color=COL[arm], lw=2.6, zorder=5)
            note = f"{med:.0f}" + ("" if len(v) == 5 else f"  ({len(v)}/5)")
            ax.text(99, y + .30, note, va="bottom", ha="right", fontsize=9.5,
                    color=INK, fontweight="700")
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([LAB[a] for a in reversed(ARMS)], fontsize=9, color=INK2)
        ax.set_xlim(0, 100); ax.set_ylim(-.65, len(ARMS) - .25)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(TITLE[cell], fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=6)
        if k // 3 == 1:
            ax.set_xlabel("final win rate (%)", fontsize=9.5, color=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.grid(axis="x", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)

    fig.text(.115, .945, "JaxNav factorial — final win rate by arm",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(.115, .903,
             "Dots are individual seeds, bar is the median. Dashed grey line = uniform replay, "
             "the baseline prioritisation must beat.",
             fontsize=9.5, color=INK2)
    fig.text(.115, .869,
             f"Win rate = mean of each run's last {WINDOW} evaluations (a single eval is 100 "
             "episodes and swings by 20+ points).",
             fontsize=9.5, color=INK2)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
