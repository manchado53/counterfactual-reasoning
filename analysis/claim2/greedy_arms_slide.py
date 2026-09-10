"""Slide figure: the JaxNav coverage experiment — curves plus a legend-table of every arm.

Self-contained for a slide: the curve panel and the arm definitions sit in one image, with
the line style and colour of each arm repeated in its table row so the two read together.

Curves come from analysis/claim2/greedy_learning_curves.py (same collect/smoothing, so the
two figures cannot disagree).
"""
import os

import numpy as np

import greedy_learning_curves as G   # same directory

OUT = os.path.join(G.WT, "docs/figures/real/claim2/jaxnav")

# key, label, colour, linestyle, lw, rollout, metric, agg, n_score, interval, coverage, purpose
ARMS = [
    ("uniform", "uniform", "#5a5a5a", "-", 2.0,
     "—", "—", "—", "—", "—", "—",
     "no priority at all — pick slots at random"),
    ("per", "PER", "#e07b39", ":", 2.2,
     "—", "—", "—", "—", "—", "—",
     "TD error only — the standard method"),
    ("cce_base_bs", "CCE shipped", "#1f6fb2", "-", 2.4,
     "T=0.5  ×20", "TV", "wmean", "64", "250", "0.86%",
     "what the PAPER describes — noisy rollouts"),
    ("greedy_base_max", "greedy_base max", "#2ca02c", "-", 2.2,
     "T=0  ×1", "W", "max", "64", "250", "0.86%",
     "REFERENCE — new recipe, OLD coverage"),
    ("greedy_base_wm", "greedy_base wmean", "#2ca02c", "--", 2.2,
     "T=0  ×1", "W", "wmean", "64", "250", "0.86%",
     "same, other aggregation — isolates the recipe"),
    ("greedy_cov_max", "greedy_cov max", "#c0392b", "-", 2.7,
     "T=0  ×1", "W", "max", "1,280", "250", "~16%",
     "THE EXPERIMENT — coverage via BIGGER BATCHES"),
    ("greedy_cov_wm", "greedy_cov wmean", "#c0392b", "--", 2.7,
     "T=0  ×1", "W", "wmean", "1,280", "250", "~16%",
     "same, other aggregation."),
    ("greedy_freq_max", "greedy_freq max", "#8e44ad", "-", 2.5,
     "T=0  ×1", "W", "max", "128", "25", "~16%",
     "same coverage via MORE OFTEN (128/25 = 1280/250)"),
]
COLS = [("rollout", .300), ("metric", .400), ("agg", .467),
        ("n_score", .553), ("interval", .630), ("coverage", .715)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SEEDS = set(range(5))       # EVERY arm restricted to seeds 0-4
    curves = G.collect(seeds=SEEDS)
    fig = plt.figure(figsize=(15, 11.6), dpi=150)
    ax = fig.add_axes([.062, .455, .915, .455])
    tb = fig.add_axes([.0, .0, 1, .40])
    tb.set_xlim(0, 1); tb.set_ylim(0, 1); tb.axis("off")   # data coords == axes fraction

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
    ax.set_xlabel("episode", fontsize=11)
    ax.set_ylabel("win rate  (%)", fontsize=11)
    ax.grid(alpha=.18, lw=.7)
    ax.set_title("JaxNav 8×8, fill 0.3 — win rate over training   (rolling median over 5 evals, "
                 "median across seeds)\nEVERY ARM: SEEDS 0-4 ONLY, one run per seed — all eight "
                 "medians are over the same 5 runs.  Shaded band = uniform's IQR.  "
                 "Lines stop where the runs currently are.",
                 fontsize=11, loc="left", pad=10)

    # ── table ────────────────────────────────────────────────────────────────
    y = .935
    tb.text(.062, y + .045, "THE ARMS", fontsize=13, weight="bold")
    tb.text(.062, y, "arm", fontsize=9.5, weight="bold", color="#55666f")
    for name, x in COLS:
        tb.text(x, y, name, fontsize=9.5, weight="bold", color="#55666f", ha="center")
    tb.text(.760, y, "what it's for", fontsize=9.5, weight="bold", color="#55666f")
    tb.plot([.062, .995], [y - .028, y - .028], color="#c5cfd6", lw=1,
            transform=tb.transAxes, clip_on=False)

    row = y - .085
    for key, lab, colour, ls, lw, *cells in ARMS:
        c = curves.get(key)
        n = f"n={c['n']}" if c else "—"
        tb.add_line(Line2D([.068, .105], [row + .012, row + .012], color=colour, ls=ls,
                           lw=lw + .4, clip_on=False))
        tb.text(.115, row, f"{lab}  {n}", fontsize=10, color=colour, weight="bold")
        for (name, x), val in zip(COLS, cells[:6]):
            tb.text(x, row, val, fontsize=9.5, ha="center", color="#101a22",
                    family="DejaVu Sans Mono")
        tb.text(.760, row, cells[6], fontsize=9.5, color="#55666f")
        row -= .098

    tb.plot([.062, .977], [row + .062, row + .062], color="#c5cfd6", lw=1,
            transform=tb.transAxes, clip_on=False)
    tb.text(.062, row - .01,
            "All arms use SEEDS 0-4 ONLY. This equalises NOISE, not worlds — arms consume RNG "
            "differently, so seed s is a different environment sequence in each.\n"
            "Every green and purple arm shares one recipe: GREEDY rollouts (T=0), ONE rollout per "
            "action, Wasserstein, bootstrap ON, buffer 100,000.  One rollout is EXACT there — the "
            "env and the policy are both deterministic, so 20 rollouts would be the same rollout 20 "
            "times, and that 20× saving is what buys the coverage at identical GPU cost.\n"
            "T = rollout temperature · ×N = rollouts per action · W = Wasserstein · TV = total "
            "variation · coverage = share of the buffer holding a MEASURED score.",
            fontsize=9.2, color="#55666f", va="top")
    p = os.path.join(OUT, "greedy_arms_slide.png")
    fig.savefig(p, facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
