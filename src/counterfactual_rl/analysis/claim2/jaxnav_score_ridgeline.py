"""Snapshots of the buffer's CCE score distribution at a handful of checkpoints.

The same `score_hist` rows that make the heatmap (`jaxnav_score_distribution.py`)
also make this, which is the slide version: a few distributions stacked, so the
shape at each moment is readable instead of being compressed into a colour.

Both arms are drawn at every checkpoint because the interesting thing is that the
spike TRAVELS without ever spreading -- bootstrap moves it about 40x to the right
and the buffer is just as concentrated when it arrives.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_score_ridgeline [--n 6]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import RUNS, ROOT, resolve

MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_instrument.json")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_score_ridgeline.png")

C_OFF, C_ON = "#0072b2", "#d55e00"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
ARMS = [("cce_wmean", "bootstrap OFF", C_OFF), ("cce_wmean_bs", "bootstrap ON", C_ON)]


def load(array_id, arm):
    man = json.load(open(MANIFEST))
    res = resolve(array_id)
    best = []
    for rec in man.values():
        if rec["arm"] != arm or rec["job_id"] not in res:
            continue
        p = os.path.join(RUNS, res[rec["job_id"]][0], "ess.jsonl")
        if os.path.exists(p):
            r = [json.loads(l) for l in open(p) if l.strip() and "score_hist" in l]
            if len(r) > len(best):
                best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274678")
    ap.add_argument("--n", type=int, default=6, help="how many checkpoints to draw")
    args = ap.parse_args()

    series = {a: load(args.array, a) for a, _, _ in ARMS}
    if not all(series.values()):
        print("not enough data yet")
        return
    floor = min(r[-1]["episode"] for r in series.values())
    r0 = series["cce_wmean"][-1]
    lo, hi, nb = r0["score_hist_lo"], r0["score_hist_hi"], r0["score_hist_nbins"]
    edges = np.logspace(np.log10(lo), np.log10(hi), nb + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # checkpoints evenly spaced in episodes, first one early enough to catch the
    # zero-mass phase before it dies out (~3k episodes)
    targets = np.linspace(floor / args.n, floor, args.n)

    def row_at(arm, ep):
        rows = [x for x in series[arm] if x["episode"] <= ep]
        return rows[-1] if rows else None

    fig, ax = plt.subplots(figsize=(13.5, 8.6), facecolor=SURFACE)
    fig.subplots_adjust(left=.135, right=.80, top=.775, bottom=.085)
    ax.set_facecolor(SURFACE)

    STEP = 1.0
    for i, ep in enumerate(targets):
        base = i * STEP
        ax.axhline(base, color=GRID, lw=.9, zorder=1)
        for arm, lab, col in ARMS:
            r = row_at(arm, ep)
            if r is None:
                continue
            h = np.array(r["score_hist"], float) / r["n"]
            y = base + h / max(h.max(), 1e-12) * STEP * 0.86
            # Plot EVERY bin including the empty ones. Drawing only the occupied
            # bins connects the line across the gaps between them, which paints a
            # broad distribution where the buffer actually has two spikes and
            # nothing in between -- the mass is 2 bins wide, not 2 decades.
            ax.fill_between(centers, base, y, step="mid", color=col, alpha=.30,
                            lw=0, zorder=2 + i)
            ax.step(centers, y, where="mid", color=col, lw=1.9, zorder=2 + i,
                    label=lab if i == 0 else None)
            # where the spike is, and how much it holds
            k = int(h.argmax())
            ax.plot([centers[k]], [base + STEP * .90], marker="v", ms=6, color=col,
                    zorder=40)
            if i == len(targets) - 1:
                ax.text(centers[k], base + STEP * 1.00, f"{100*h[k]:.0f}%",
                        ha="center", fontsize=9.5, color=col, fontweight="bold")
        rr = row_at("cce_wmean", ep)
        ax.text(lo * 0.55, base + STEP * .30,
                f"{rr['episode']/1000:,.0f}k\nep",
                ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(lo * 0.55, base - STEP * .09,
                f"{100*rr['score_n_zero']/rr['n']:.1f}% at 0",
                ha="right", va="center", fontsize=7.8, color=INK2)

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-.35, len(targets) * STEP + .55)
    ax.set_yticks([])
    ax.set_xlabel("CCE score stored in the buffer   (log scale)", fontsize=11, color=INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", alpha=.5, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left",
              bbox_to_anchor=(1.005, 1.0))
    ax.text(1.005, .80,
            "each curve is normalised\nto its own peak, so the\nSHAPE is comparable\n"
            "across rows.\n\n▼ marks the tallest bin;\nthe % on the top row is\n"
            "what that one bin holds.",
            transform=ax.transAxes, fontsize=9, color=INK2, va="top")

    fig.text(.055, .955, "The buffer's score distribution, checkpoint by checkpoint",
             fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(.055, .908,
             f"JaxNav 8x8 fill 0.3  ·  array {args.array}  ·  cce_wmean, identical "
             f"except cf_bootstrap  ·  the seed furthest along  ·  up to "
             f"{floor:,} of 250,000 episodes.\n\n"
             "The spike TRAVELS and never spreads. Bootstrap moves it ~40x to the "
             "right and the buffer is exactly as concentrated when it gets there —\n"
             "that single bin is the running mean every unscored entry inherits, and "
             "it is ~99% of the buffer in every row.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.55)

    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
