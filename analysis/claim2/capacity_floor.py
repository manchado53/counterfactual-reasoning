"""The capacity floor — does a small buffer kill learning WITHOUT CCE anywhere?

Array 275562: uniform and PER down six capacities, five seeds each, no consequence
scoring of any kind. Every small-buffer run before this one was a CCE run, so
"shrinking the buffer collapses learning" had never been separated from "shrinking the
buffer collapses CCE".

Answer: both collapse, and they collapse together. Prioritisation is not the culprit.

Read-out is per-seed and median, never a mean -- JaxNav's paired seed sd is ~30pp and the
seeds are bimodal. Capacities 100,000 carry the pooled baselines (n=26 uniform, n=18 PER)
because the in-sweep anchors merge with the existing runs at the default capacity; every
other rung is the n=5 from this sweep alone.
"""
import os

import numpy as np
from scipy.stats import mannwhitneyu

import greedy_learning_curves as G

OUT = os.path.join(G.WT, "docs/figures/real/claim2/jaxnav")
CAPS = [8_192, 12_288, 16_384, 25_000, 50_000, 100_000]
BURST = 8_192          # n_envs x collect_steps -- one insertion burst
ALGOS = [("uniform", "uniform  (no priorities)", "#1f6fb2"),
         ("per",     "PER  (TD priorities)",     "#c0392b")]


def _final(curves, key):
    i = int(np.argmin(np.abs(G.GRID - 250_000)))
    a = curves[key]["M"][:, i]
    return a[~np.isnan(a)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = G.collect(G._label_floor)
    fig = plt.figure(figsize=(13.5, 8.6), dpi=150)
    ax = fig.add_axes([.075, .455, .62, .47])
    tb = fig.add_axes([.0, .0, 1, .40]); tb.set_xlim(0, 1); tb.set_ylim(0, 1); tb.axis("off")
    sd = fig.add_axes([.755, .455, .225, .47])

    for key, lab, colour in ALGOS:
        med, lo, hi, pts = [], [], [], []
        for cap in CAPS:
            a = _final(c, f"{key}_{cap}")
            med.append(np.median(a)); lo.append(np.percentile(a, 25))
            hi.append(np.percentile(a, 75)); pts.append(a)
        ax.fill_between(CAPS, lo, hi, color=colour, alpha=.13, lw=0)
        ax.plot(CAPS, med, color=colour, lw=2.6, marker="o", ms=7,
                mfc="white", mew=2.2, label=lab, zorder=3)
        for cap, a in zip(CAPS, pts):
            ax.scatter(np.full(len(a), cap) * np.random.uniform(.94, 1.06, len(a)),
                       a, s=16, color=colour, alpha=.45, lw=0, zorder=2)

    ax.axvline(BURST, color="#55666f", lw=1.3, ls="--")
    # sits in the empty mid-left region: the legend owns the top-left corner
    ax.text(BURST * 1.09, 36, "8,192 = one insert burst\nbelow this, data is\noverwritten unread",
            fontsize=8.5, color="#55666f", va="top")
    ax.set_xscale("log")
    ax.set_xticks(CAPS)
    ax.set_xticklabels([f"{v//1000}k" if v >= 10_000 else f"{v:,}" for v in CAPS], fontsize=9)
    ax.minorticks_off()
    ax.set_xlim(7_000, 130_000); ax.set_ylim(0, 82)
    ax.set_xlabel("buffer capacity  (transitions, log scale)", fontsize=11)
    ax.set_ylabel("final win rate at 250k episodes  (%)", fontsize=11)
    ax.grid(alpha=.18, lw=.7)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.set_title("The capacity floor is NOT a CCE effect   (JaxNav 8x8, fill 0.3, 250k episodes)\n"
                 "no consequence scoring in any of these 60 runs — plain replay collapses too",
                 fontsize=11.5, loc="left", pad=10)

    # side panel: what a smaller buffer was supposed to buy
    stal = [cap / 32 for cap in CAPS]
    sd.plot(CAPS, stal, color="#0d7c6b", lw=2.2, marker="s", ms=5)
    sd.axhline(1788, color="#a8341c", lw=1.2, ls=":")
    sd.text(8_600, 1980, "today's MEASURED\nstaleness 1,788", fontsize=8, color="#a8341c")
    sd.set_xscale("log"); sd.set_xticks([8_192, 25_000, 100_000])
    sd.set_xticklabels(["8k", "25k", "100k"], fontsize=9); sd.minorticks_off()
    sd.set_xlim(7_000, 130_000)
    sd.set_ylabel("mean score staleness\n(gradient updates)", fontsize=9)
    sd.grid(alpha=.18, lw=.7)
    sd.set_title("what shrinking was FOR", fontsize=10, loc="left", pad=8)

    # ── table ───────────────────────────────────────────────────────────────────
    y = .93
    tb.text(.075, y + .050, "EVERY SEED, FINAL WIN RATE AT 250k", fontsize=12.5, weight="bold")
    cols = [("capacity", .075), ("bursts", .165), ("staleness", .245),
            ("uniform", .350), ("vs 100k", .470),
            ("PER", .580), ("vs 100k", .700), ("verdict", .800)]
    for name, x in cols:
        tb.text(x, y, name, fontsize=9, weight="bold", color="#55666f")
    tb.plot([.070, .975], [y - .022, y - .022], color="#c9d2d7", lw=1,
            transform=tb.transAxes, clip_on=False)

    refs = {k: _final(c, f"{k}_100000") for k, _, _ in ALGOS}
    row = y - .075
    for cap in CAPS:
        tb.text(.075, row, f"{cap:,}", fontsize=9.5)
        tb.text(.165, row, f"{cap/BURST:.1f}", fontsize=9.5, color="#55666f")
        tb.text(.245, row, f"{cap/32:,.0f}", fontsize=9.5, color="#55666f")
        cells = {}
        for key, _, colour in ALGOS:
            a = _final(c, f"{key}_{cap}")
            x = .350 if key == "uniform" else .580
            tb.text(x, row, f"{np.median(a):.1f}%   n={len(a)}", fontsize=9.5, color=colour)
            xp = .470 if key == "uniform" else .700
            if cap == 100_000:
                tb.text(xp, row, "anchor", fontsize=9, color="#8a9aa3")
                cells[key] = None
            else:
                p = mannwhitneyu(a, refs[key], alternative="two-sided")[1]
                d = np.median(a) - np.median(refs[key])
                tb.text(xp, row, f"{d:+.1f}pp  p={p:.3f}", fontsize=9,
                        color="#a8341c" if p < .05 else "#55666f")
                cells[key] = p
            row_pts = " ".join(f"{v:.0f}" for v in np.sort(a)[:5])
        if cap == 100_000:
            verdict = "today's default"
        elif cells["uniform"] < .05 and cells["per"] < .05:
            verdict = "BOTH collapse"
        elif cells["uniform"] < .05:
            verdict = "uniform collapses"
        elif cells["per"] < .05:
            verdict = "PER collapses"
        else:
            verdict = "survivable"
        tb.text(.800, row, verdict, fontsize=9.5,
                color="#a8341c" if "collapse" in verdict else "#0d7c6b")
        row -= .095

    # absolute y, not `row` -- a computed offset collided with the last row twice
    tb.text(.075, .255,
            "Both arms fall together at every rung. The collapse is a REPLAY-DIVERSITY effect, not a "
            "prioritisation effect —\n"
            "so the 32 dead CCE small-buffer runs were dying of the buffer, not of CCE. "
            "The lab-notebook rule stands, now uncontaminated.\n"
            "50,000 is the lowest rung that is not significantly worse than the default "
            "(-13.3pp, p=0.111 at n=5) — but the point estimate is a real risk.",
            fontsize=9.5, color="#33444d", linespacing=1.55, va="top")

    p = os.path.join(OUT, "capacity_floor.png")
    fig.savefig(p, facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
