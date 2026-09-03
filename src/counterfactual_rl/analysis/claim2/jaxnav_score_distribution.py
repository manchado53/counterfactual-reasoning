"""Issue #8 — the distribution of CCE scores ACROSS THE REPLAY BUFFER, over training.

Jeremy asked for a KDE of the buffer's scores. This is that, as a trajectory rather
than a single snapshot, which the logging now makes free: `priority_diagnostics`
writes a 100-bin log-spaced histogram (plus separate exactly-zero / below-range /
above-range counters) at EVERY eval, so one run yields the paper heatmap, a slide
ridgeline and a talk animation with no re-run.

WHY LOG BINS AND A SEPARATE ZERO COUNTER. `score_mean` travels ~0.0014 -> ~0.075 over
a run, so linear bins put every entry in bin 0 early and bin 7 late. A log axis cannot
represent 0, and "exactly zero" is a real and distinct category, so it is carried
alongside rather than folded into the first bin.

WHAT THIS SHOWS THAT NOTHING BEFORE COULD. `jaxnav_signal_birth.py` plots scores of
states sampled ON-POLICY at checkpoints -- not the scores STORED in the buffer, which
is what the sampler actually draws on. The buffer was never checkpointed, so the
stored distribution could not be recovered after the fact. This is that distribution.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_score_distribution [--array 274678]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import RUNS, ROOT, resolve

MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_instrument.json")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_score_distribution.png")

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
HOT = "#d55e00"
ARM = "cce_wmean"


def load_arm(array_id, arm):
    man = json.load(open(MANIFEST))
    res = resolve(array_id)
    runs = []
    for rec in man.values():
        if rec["arm"] != arm or rec["job_id"] not in res:
            continue
        p = os.path.join(RUNS, res[rec["job_id"]][0], "ess.jsonl")
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p) if l.strip()]
            if rows and "score_hist" in rows[-1]:
                runs.append(rows)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274678")
    ap.add_argument("--arm", default=ARM)
    args = ap.parse_args()

    runs = load_arm(args.array, args.arm)
    if not runs:
        print("no histogram data yet")
        return
    rows = max(runs, key=len)          # the seed that has progressed furthest
    r0 = rows[-1]
    lo, hi, nb = r0["score_hist_lo"], r0["score_hist_hi"], r0["score_hist_nbins"]
    edges = np.logspace(np.log10(lo), np.log10(hi), nb + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    eps = np.array([r["episode"] for r in rows], float)
    H = np.vstack([np.array(r["score_hist"], float) / r["n"] for r in rows]).T   # (bins, time)
    zero = np.array([r["score_n_zero"] / r["n"] for r in rows], float)
    print(f"arm {args.arm}: {len(rows)} evals, up to episode {eps[-1]:,.0f}")

    fig = plt.figure(figsize=(15.5, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1], height_ratios=[7, 1.25],
                          left=.062, right=.985, top=.775, bottom=.105,
                          wspace=.20, hspace=.10)

    # ---- heatmap: distribution over training ----------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    vmin = 1.0 / r0["n"]
    # Sequential = ONE hue, light -> dark. Log norm because a single bin holds ~70%
    # of the buffer and a linear ramp would paint every other bin identical white.
    m = ax.pcolormesh(eps, centers, np.ma.masked_less(H, vmin), cmap="Blues",
                      norm=mcolors.LogNorm(vmin=vmin, vmax=1.0), shading="nearest")
    ax.set_yscale("log")
    ax.set_ylabel("CCE score stored in the buffer", fontsize=10, color=INK2)
    ax.set_title(f"1.  Where the buffer's scores actually sit  ({args.arm})",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=9)
    ax.tick_params(colors=INK2, labelsize=9, labelbottom=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cb = fig.colorbar(m, ax=ax, pad=.012, fraction=.037)
    cb.set_label("share of the buffer", fontsize=9, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)

    band = H.max(axis=0)
    ax.annotate(f"one bin holds {100*band[-1]:.0f}% of the buffer\n"
                f"= the running mean every unscored entry inherits",
                xy=(eps[-1], centers[H[:, -1].argmax()]),
                xytext=(eps[-1] * .30, 3e-2),
                fontsize=9.5, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=HOT, lw=1.5))

    # ---- the zero mass, its own strip (a log axis cannot hold 0) --------
    axz = fig.add_subplot(gs[1, 0], sharex=ax)
    axz.set_facecolor(SURFACE)
    axz.fill_between(eps, 0, 100 * zero, color=HOT, alpha=.85, lw=0)
    axz.set_ylabel("= 0\n(%)", fontsize=8.5, color=INK2)
    axz.set_xlabel("episodes (thousands)", fontsize=10, color=INK2)
    axz.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
    axz.tick_params(colors=INK2, labelsize=8.5)
    axz.set_ylim(0, max(1.0, 100 * zero.max() * 1.3))
    for sp in ("top", "right"):
        axz.spines[sp].set_visible(False)
    axz.text(.012, .58, "scores of EXACTLY zero — carried separately, a log axis has no 0",
             transform=axz.transAxes, fontsize=8, color=INK2)

    # ---- latest snapshot -------------------------------------------------
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.set_facecolor(SURFACE)
    h = H[:, -1]
    occ = h > 0
    ax2.step(centers[occ], 100 * h[occ], where="mid", color="#0072b2", lw=2)
    ax2.fill_between(centers[occ], 0, 100 * h[occ], step="mid",
                     color="#0072b2", alpha=.22)
    ax2.set_xscale("log")
    ax2.set_xlabel("CCE score", fontsize=10, color=INK2)
    ax2.set_ylabel("% of the buffer", fontsize=10, color=INK2)
    ax2.set_title(f"2.  The same thing at episode {eps[-1]:,.0f}", fontsize=12,
                  fontweight="bold", color=INK, loc="left", pad=9)
    ax2.grid(alpha=.55, color=GRID, lw=.8)
    ax2.set_axisbelow(True)
    ax2.tick_params(colors=INK2, labelsize=9)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    n_occ = int(occ.sum())
    top2 = 100 * np.sort(h)[::-1][:2].sum()
    ax2.text(.03, .93,
             f"{top2:.1f}% of the buffer in 2 adjacent bins\n"
             f"only {n_occ} of {nb} bins occupied at all\n"
             f"{100*zero[-1]:.2f}% score exactly 0",
             transform=ax2.transAxes, fontsize=10, color=INK,
             va="top", fontweight="bold")

    fig.text(.062, .935, "The replay buffer's score distribution  (issue #8)",
             fontsize=18, fontweight="bold", color=INK)
    fig.text(.062, .888,
             f"JaxNav 8×8 fill 0.3 · array {args.array} · arm {args.arm} · "
             f"the seed furthest along · 100 log-spaced bins logged every eval.",
             fontsize=10.5, color=INK2)
    fig.text(.062, .850,
             "This is the distribution STORED IN THE BUFFER — what the sampler draws on. "
             "Earlier figures scored states sampled on-policy, which is a different thing; "
             "the buffer was never saved, so this could not be recovered after the fact.",
             fontsize=9.5, color=INK2, style="italic")

    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
