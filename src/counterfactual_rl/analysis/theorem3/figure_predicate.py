"""Slide figure — the predicate separates deterministic from slippery FrozenLake.

One panel, one claim: how often does the CCE side of Theorem 3's predicate beat
the TD side, per converged run, in each environment.

An earlier two-panel version also claimed "both sides swap sign between
environments", drawn from medians over 7 and 8 runs. With 13 and 10 runs the TD
side's median in deterministic FL flips from -1.5e-02 to +1.0e-02, so that claim
was a small-sample artifact and the panel carrying it has been removed.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from counterfactual_rl.analysis.theorem3.priority_flatness import (
    OUT_DIR, BLUE, ORANGE, INK, INK2, GRID,
)

FILES = (("Deterministic  (slip 0)\nwhere CCE+TD won 1.0 vs 0.45",
          "step2_graded_slip0.0_dqn-uniform.json", BLUE),
         ("Full slip  (slip 0.667)\nwhere Claim 2 is null",
          "step2_graded_slip0.666_dqn-uniform.json", ORANGE))


def load(fn, agg="max"):
    """Converged, winning checkpoints only — a diverged net makes |Q-Q*| meaningless."""
    recs = json.load(open(os.path.join(OUT_DIR, fn)))
    good = [r for r in recs
            if r["global_err"] < 1.0 and r.get("achieved_win_rate", 0) > 0.5
            and r.get(f"{agg}_predicate")]
    lhs = np.array([r[f"{agg}_predicate"]["lhs"] for r in good])
    rhs = np.array([r[f"{agg}_predicate"]["rhs"] for r in good])
    return lhs, rhs


def main(agg="max"):
    data = [(lab, col) + load(fn, agg) for lab, fn, col in FILES]
    (_, _, l0, r0), (_, _, l1, r1) = data
    w0, w1 = int((l0 >= r0).sum()), int((l1 >= r1).sum())
    pval = fisher_exact([[w0, len(l0) - w0], [w1, len(l1) - w1]],
                        alternative="greater")[1]

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    fig.subplots_adjust(left=0.155, right=0.965, top=0.70, bottom=0.16)

    rng = np.random.default_rng(0)
    for i, (lab, col, lhs, rhs) in enumerate(data):
        m = lhs - rhs
        x = i + (rng.random(len(m)) - 0.5) * 0.22
        ax.scatter(x, m, s=105, color=col, alpha=0.9, lw=1.6,
                   edgecolor="white", zorder=3)
        ax.plot([i - 0.27, i + 0.27], [np.median(m)] * 2, lw=2.8, color=col, zorder=4)
        wins = int((m >= 0).sum())
        ax.text(i, 1.03, f"{wins} of {len(m)}", ha="center", va="bottom",
                fontsize=15, color=col, fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.text(i, 1.005, "runs where CCE is the better signal", ha="center",
                va="bottom", fontsize=9.5, color=INK2,
                transform=ax.get_xaxis_transform())

    ax.axhline(0, color=INK2, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.set_yscale("symlog", linthresh=1e-4)
    span = max(np.abs(np.concatenate([l0 - r0, l1 - r1]))) * 3.2   # headroom for labels
    ax.set_ylim(-span, span)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=11.5, color=INK2)
    ax.set_xlim(-0.62, len(data) - 0.38)
    ax.set_ylabel("Cov(c,u)/E[c]  −  Cov(d,u)/E[d]\n← TD better      CCE better →",
                  color=INK2, fontsize=11)
    ax.text(-0.6, -span * 0.72, "each dot = one converged run · bar = median",
            fontsize=9.5, color=INK2, va="center", ha="left")

    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)

    fig.suptitle("Under slip, CCE's signal never beats TD error",
                 fontsize=17, fontweight="bold", color=INK, x=0.155, ha="left", y=0.955)
    fig.text(0.155, 0.845,
             "FrozenLake 8×8 · neutral DQN-uniform checkpoints, matched by achieved win rate\n"
             f"{agg} aggregation · Fisher one-sided p = {pval:.3f}",
             fontsize=10.5, color=INK2, ha="left", va="top")

    path = os.path.join(OUT_DIR, "fig_predicate_by_env.png")
    fig.savefig(path, dpi=170, facecolor="white")
    print("wrote", path)
    for (lab, col, lhs, rhs), w in zip(data, (w0, w1)):
        print(f"  {lab.splitlines()[0]:>26}: n={len(lhs):2d}  CCE wins {w}/{len(lhs)}  "
              f"LHS med {np.median(lhs):+.2e}  RHS med {np.median(rhs):+.2e}")
    print(f"  Fisher one-sided p = {pval:.4f}")


if __name__ == "__main__":
    main()
