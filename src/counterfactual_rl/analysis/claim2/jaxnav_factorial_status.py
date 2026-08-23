"""Status board for the 6-cell JaxNav factorial (array 273775).

One panel per cell of the size x fill grid, one bar per arm, filled by mean
progress across that arm's seeds. Existing to answer "what is done, what is
running, what has not started" at a glance while a multi-day sweep is in flight.

Reads run directories via the manifest's `run_dir` field, NOT `job_id`: a SLURM
array task addressed as <array>_<task> is allocated a distinct JobIDRaw, and the
trainer names its run directory after that raw id. Run slurm/resolve_manifest.py
first to populate it.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_factorial.json")
RUNS = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_factorial_status.png")

TARGET = 250_000
FILLS = [0.1, 0.3, 0.5]
SIZES = [(8, 8), (11, 11)]
ARMS = ["uniform", "per", "cce_wmean", "cce_max", "cce_add", "cce_only"]
LABEL = {"uniform": "uniform", "per": "PER", "cce_wmean": "CCE mul·wmean",
         "cce_max": "CCE mul·max", "cce_add": "CCE add μ=.25", "cce_only": "CCE only μ=1"}

INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"
DONE, LIVE, IDLE = "#2f7d4f", "#0072b2", "#e3e0da"


def progress():
    man = json.load(open(MANIFEST))
    out = {}
    for rec in man.values():
        ep, wr = 0, None
        rd = rec.get("run_dir")
        if rd:
            f = os.path.join(RUNS, str(rd), "metrics.log")
            if os.path.exists(f):
                rows = [l.split() for l in open(f)
                        if not l.startswith("#") and l.strip()]
                rows = [r for r in rows if len(r) >= 5 and r[0] != "episode"]
                if rows:
                    ep = int(float(rows[-1][0]))
                    wr = float(rows[-1][3].rstrip("%"))
        out.setdefault((rec["cell"], rec["arm"]), []).append((ep, wr))
    return out


def main():
    prog = progress()
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4))
    fig.subplots_adjust(left=.13, right=.98, top=.80, bottom=.05, hspace=.40, wspace=.16)

    total_ep = total_target = 0
    for r, size in enumerate(SIZES):
        for c, fill in enumerate(FILLS):
            cell = f"{size[0]}x{size[1]}_f{str(fill).replace('0.','0')}"
            ax = axes[r][c]
            for i, arm in enumerate(ARMS):
                y = len(ARMS) - 1 - i
                runs = prog.get((cell, arm), [])
                n = len(runs) or 5
                eps = [e for e, _ in runs] + [0] * (5 - len(runs))
                frac = np.mean([min(e / TARGET, 1.0) for e in eps])
                ndone = sum(1 for e in eps if e >= TARGET * 0.96)
                total_ep += sum(eps); total_target += 5 * TARGET
                col = DONE if ndone == 5 else (LIVE if frac > 0 else IDLE)
                ax.add_patch(Rectangle((0, y - .34), 1, .68, facecolor=IDLE,
                                       edgecolor="none"))
                if frac > 0:
                    ax.add_patch(Rectangle((0, y - .34), frac, .68, facecolor=col,
                                           edgecolor="none"))
                wrs = [w for _, w in runs if w is not None]
                note = f"{ndone}/5" if ndone else (f"{100*frac:.0f}%" if frac else "")
                if ndone == 5 and wrs:
                    note += f"   win {np.median(wrs):.0f}%"
                ax.text(1.02, y, note, va="center", ha="left", fontsize=8.5,
                        color=INK if frac else "#a9a49c",
                        fontweight="600" if ndone == 5 else "normal")
            ax.set_xlim(0, 1.45); ax.set_ylim(-.7, len(ARMS) - .3)
            ax.set_yticks(range(len(ARMS)))
            ax.set_yticklabels([LABEL[a] for a in reversed(ARMS)], fontsize=8.5, color=INK2)
            ax.set_xticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            nd = sum(1 for a in ARMS
                     for e, _ in prog.get((cell, a), []) if e >= TARGET * .96)
            ax.set_title(f"{size[0]}×{size[1]}   fill {fill}          {nd}/30 runs done",
                         fontsize=10, color=INK, fontweight="bold", loc="left", pad=7)

    pct = 100 * total_ep / max(total_target, 1)
    fig.text(.013, .955, "JaxNav 6-cell factorial — what is filled in",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(.013, .921,
             f"180 runs · 6 arms × 5 seeds × 250k episodes.   Overall {pct:.0f}% of all "
             f"episodes collected.",
             fontsize=9.5, color=INK2)
    fig.text(.013, .893,
             "Bars show mean progress across an arm's 5 seeds; win rate is the median "
             "of a completed arm.",
             fontsize=9.5, color=INK2)
    for x, (col, lab) in zip([.615, .715, .815],
                             [(DONE, "arm complete"), (LIVE, "in progress"), (IDLE, "not started")]):
        fig.patches.append(Rectangle((x, .955), .013, .019, transform=fig.transFigure,
                                     facecolor=col, edgecolor="none"))
        fig.text(x + .019, .964, lab, fontsize=9, color=INK2, va="center")
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}   ({pct:.1f}% of all episodes collected)")


if __name__ == "__main__":
    main()
