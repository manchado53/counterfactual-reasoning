"""Is the CCE score actually doing anything to the sampler?

Reads ess.jsonl, which every CCE arm writes each eval. Two questions:

  1. CONCENTRATION -- realised ESS/n. 1.0 is uniform replay. If a "CCE" arm sits
     at 1.0 it is sampling uniformly no matter what its priority formula says.
     Measured on 8x8_f01: cce_only sat at 0.999 while the mixed arms were ~0.92,
     which says the concentration comes from the TD term, not from CCE.

  2. SIGNAL HEALTH -- spread of the consequence score across the buffer
     (score_std / score_mean). A score that collapses toward one value cannot
     rank anything. Only ~1.6% of the buffer ever carries a measured score; the
     rest inherits mean(existing) on add(), so this is expected to be small --
     the question is whether it degrades further as the policy improves.

    python -m counterfactual_rl.analysis.claim2.jaxnav_signal_health
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
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_signal_health.png")

ARMS = ["cce_wmean", "cce_max", "cce_add", "cce_only"]
COL = {"cce_wmean": "#009e73", "cce_max": "#d55e00",
       "cce_add": "#bf8600", "cce_only": "#8452c4"}
LAB = {"cce_wmean": "CCE mul·wmean", "cce_max": "CCE mul·max",
       "cce_add": "CCE add μ=.25", "cce_only": "CCE only μ=1"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def load():
    man = json.load(open(MANIFEST))
    out = {}
    for rec in man.values():
        rd = rec.get("run_dir")
        if not rd or rec["arm"] not in ARMS:
            continue
        p = os.path.join(RUNS, str(rd), "ess.jsonl")
        if not os.path.exists(p):
            continue
        ep, ess, cv = [], [], []
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep.append(r["episode"]); ess.append(r["ess_frac"])
            m, s = r.get("score_mean") or 0.0, r.get("score_std") or 0.0
            cv.append(s / m if m > 1e-9 else 0.0)
        if len(ep) > 3:
            out.setdefault((rec["cell"], rec["arm"]), []).append(
                (np.array(ep), np.array(ess), np.array(cv)))
    return out


def band(ax, runs, idx, col, label):
    hi = min(r[0][-1] for r in runs)
    grid = np.linspace(250, hi, 200)
    stack = np.vstack([np.interp(grid, r[0], r[idx]) for r in runs])
    med = np.median(stack, axis=0)
    lo, up = np.percentile(stack, [25, 75], axis=0)
    ax.fill_between(grid / 1000, lo, up, color=col, alpha=.15, lw=0)
    ax.plot(grid / 1000, med, color=col, lw=2.1, label=f"{label} (n={len(runs)})")


def main():
    data = load()
    cells = sorted({c for c, _ in data})
    if not cells:
        print("no ess.jsonl data yet"); return
    fig, axes = plt.subplots(2, len(cells), figsize=(4.6 * len(cells) + 1.4, 7.4),
                             squeeze=False)
    fig.subplots_adjust(left=.075, right=.985, top=.80, bottom=.08, hspace=.30, wspace=.20)

    for j, cell in enumerate(cells):
        for i, (idx, ylab, ylim) in enumerate(
                [(1, "realised ESS / n", (0.5, 1.02)),
                 (2, "score spread  (std / mean)", (0, None))]):
            ax = axes[i][j]
            for arm in ARMS:
                runs = data.get((cell, arm))
                if runs:
                    band(ax, runs, idx, COL[arm], LAB[arm])
            if i == 0:
                ax.axhline(1.0, color=INK2, ls=":", lw=1.2)
                ax.text(.99, 1.005, "uniform replay", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=8, color=INK2, style="italic")
                ax.set_title(cell, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=6)
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
            if i == 1:
                ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8, loc="lower left")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8.5)
            ax.grid(axis="y", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)

    fig.text(.075, .945, "Is the CCE score doing anything to the sampler?",
             fontsize=15.5, fontweight="bold", color=INK)
    fig.text(.075, .903,
             "TOP: realised concentration. 1.0 means the arm is sampling UNIFORMLY regardless of "
             "its priority formula.",
             fontsize=9.5, color=INK2)
    fig.text(.075, .871,
             "BOTTOM: spread of the consequence score across the buffer. A score that collapses "
             "toward one value cannot rank anything.",
             fontsize=9.5, color=INK2)
    fig.text(.075, .839,
             "Median across seeds, shaded band = interquartile range.",
             fontsize=9, color=INK2, style="italic")
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")

    print(f"\n{'cell':<12}{'arm':<12}{'ESS end':>9}{'spread end':>12}")
    for cell in cells:
        for arm in ARMS:
            runs = data.get((cell, arm))
            if not runs:
                continue
            e = np.median([r[1][-1] for r in runs]); c = np.median([r[2][-1] for r in runs])
            print(f"{cell:<12}{arm:<12}{e:>9.3f}{c:>12.3f}")


if __name__ == "__main__":
    main()
