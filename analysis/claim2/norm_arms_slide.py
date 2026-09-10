"""Slide figure: priority normalisation (issue #16) — outcome curves plus the arm table.

The table carries the OFFLINE sampler measurements alongside the config, because that is
the part that is already settled: what each mode does to the replay distribution was
measured from real buffer snapshots (analysis/claim2/priority_normalisation.py) and needed
no GPU. The curves are the open part -- does any of it change LEARNING.
"""
import os

import numpy as np
from matplotlib.lines import Line2D

import greedy_learning_curves as G

OUT = os.path.join(G.WT, "docs/figures/real/claim2/jaxnav")

# key, label, colour, ls, lw, then the CONFIG columns, then the SAMPLER columns, purpose.
# Config is identical across the three CCE arms -- that is the point of the figure, so the
# repeated columns are deliberate. Coverage is MEASURED from scored_frac in ess.jsonl at
# the last eval of each run (mean over 5 seeds), not derived from n_score/interval.
ARMS = [
    ("uniform", "uniform", "#5a5a5a", "-", 1.9,
     "—", "—", "—", "—", "—", "—", "—",
     "—", "—", "—", "—", "—",
     "no CCE at all — the floor"),
    ("greedy_cov_max", "none  (shipped)", "#1f6fb2", "-", 2.6,
     "T=0 ×1", "60", "W", "max", "1,280", "250", "15.4%",
     "none", "0.944", "100%", "10.0×",
     "every run to date"),
    ("cov_minmax_max", "min-max", "#e07b39", "-", 2.6,
     "T=0 ×1", "60", "W", "max", "1,280", "250", "15.8%",
     "minmax", "0.985", "45.5%", "3.2×",
     "Jeremy's 08/20 ask"),
    ("cov_rank_max", "rank", "#c0392b", "-", 2.6,
     "T=0 ×1", "60", "W", "max", "1,280", "250", "15.9%",
     "rank", "0.930", "33.2%", "1.5×",
     "issue #16 — cv 47.4 → 0.58"),
]
CFG_COLS = [("rollout", .240), ("horizon", .296), ("metric", .347), ("agg", .396),
            ("n_score", .452), ("interval", .512), ("coverage", .570)]
SMP_COLS = [("normalise", .640), ("ESS", .706), ("top-1% kept", .772),
            ("peak/unif", .845)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = G.collect(G._label_norm)
    fig = plt.figure(figsize=(17, 9.8), dpi=150)
    ax = fig.add_axes([.062, .445, .915, .470])
    tb = fig.add_axes([.0, .0, 1, .39]); tb.set_xlim(0, 1); tb.set_ylim(0, 1); tb.axis("off")

    u = curves.get("uniform")
    if u is not None:
        ax.fill_between(G.GRID, u["lo"], u["hi"], color="#5a5a5a", alpha=.13, lw=0)
    for key, lab, colour, ls, lw, *_ in ARMS:
        c = curves.get(key)
        if c is None:
            continue
        ax.plot(G.GRID, c["med"], color=colour, ls=ls, lw=lw)
    ax.axvline(G.EPS_DECAY, color="#a8341c", lw=1.2, ls="--")
    ax.text(G.EPS_DECAY + 3500, 2.5, "epsilon decay ends\nleft of here = exploration",
            fontsize=9, color="#a8341c", va="bottom")
    ax.set_xlim(0, 250_000); ax.set_ylim(0, 80)
    ax.set_xlabel("episode", fontsize=11); ax.set_ylabel("win rate  (%)", fontsize=11)
    ax.grid(alpha=.18, lw=.7)
    ax.set_title("Priority normalisation at 16% coverage — issue #16   (JaxNav 8×8, fill 0.3)\n"
                 "the three CCE arms are IDENTICAL except normalise_priorities.  "
                 "STILL RUNNING — lines stop where the runs are.",
                 fontsize=11, loc="left", pad=10)

    y = .93
    tb.text(.062, y + .052, "THE ARMS", fontsize=13, weight="bold")
    tb.text(.262, y + .052, "CONFIG  (identical across the three CCE arms)",
            fontsize=9, color="#55666f")
    tb.text(.640, y + .052, "WHAT IT DOES TO THE SAMPLER  (measured offline)",
            fontsize=9, color="#55666f")
    tb.text(.062, y, "arm", fontsize=9, weight="bold", color="#55666f")
    for name, x in CFG_COLS + SMP_COLS:
        tb.text(x, y, name, fontsize=9, weight="bold", color="#55666f", ha="center")
    tb.text(.884, y, "what it's for", fontsize=9, weight="bold", color="#55666f")
    tb.plot([.062, .995], [y - .032, y - .032], color="#c5cfd6", lw=1, clip_on=False)
    tb.plot([.607, .607], [y + .04, y - .52], color="#c5cfd6", lw=1, clip_on=False)

    row = y - .105
    for key, lab, colour, ls, lw, *cells in ARMS:
        c = curves.get(key)
        n = f"n={c['n']}" if c else "—"
        reach = f"  {c['reach']/1000:.0f}k" if c and c["reach"] < 245_000 else ""
        tb.add_line(Line2D([.066, .100], [row + .013, row + .013], color=colour, ls=ls,
                           lw=lw + .4, clip_on=False))
        tb.text(.108, row, f"{lab} {n}{reach}", fontsize=9.5, color=colour, weight="bold")
        for (name, x), val in zip(CFG_COLS + SMP_COLS, cells[:11]):
            tb.text(x, row, val, fontsize=8.8, ha="center", color="#101a22",
                    family="DejaVu Sans Mono")
        tb.text(.884, row, cells[11], fontsize=8.8, color="#55666f")
        row -= .125

    tb.plot([.062, .995], [row + .075, row + .075], color="#c5cfd6", lw=1, clip_on=False)
    tb.text(.062, row + .050,
            "COVERAGE is MEASURED from scored_frac at the last eval, mean over 5 seeds — not "
            "derived from n_score/interval.  All three CCE arms sit at ~15.5%, as intended: they "
            "differ ONLY in normalise_priorities.\n"
            "ESS / top-1% / peak come from a real buffer snapshot — a snapshot carries the scores, "
            "the TD magnitudes and scored_at, which is everything the priority computation "
            "consumes, so no GPU was needed.  ESS 1.0 = uniform replay.\n"
            "NOTE issue #16's title says min-max is a near-no-op. It is NOT — it cuts the peak from "
            "10.0× to 3.2× and reshuffles 54% of the top slots.  But every mode sits at ESS "
            "0.93–0.99, so replay stays close to uniform whatever we pick.",
            fontsize=8.6, color="#55666f", va="top")
    p = os.path.join(OUT, "norm_arms_slide.png")
    fig.savefig(p, facecolor="white"); print("wrote", p)


if __name__ == "__main__":
    main()
