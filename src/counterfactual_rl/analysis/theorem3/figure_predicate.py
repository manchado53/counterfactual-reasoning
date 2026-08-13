"""Slide figure — the predicate separates deterministic from slippery FrozenLake."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from counterfactual_rl.analysis.theorem3.priority_flatness import (
    OUT_DIR, BLUE, ORANGE, INK, INK2, GRID,
)

FILES = (("Deterministic\n(slip 0) — CCE won 1.0 vs 0.45",
          "step2_graded_slip0.0_dqn-uniform.json", BLUE),
         ("Full slip\n(slip 0.667) — the null",
          "step2_graded_slip0.666_dqn-uniform.json", ORANGE))


def load(fn, agg="max"):
    recs = json.load(open(os.path.join(OUT_DIR, fn)))
    good = [r for r in recs
            if r["global_err"] < 1.0 and r.get("achieved_win_rate", 0) > 0.5
            and r.get(f"{agg}_predicate")]
    lhs = np.array([r[f"{agg}_predicate"]["lhs"] for r in good])
    rhs = np.array([r[f"{agg}_predicate"]["rhs"] for r in good])
    return lhs, rhs


def main(agg="max"):
    data = [(lab, col) + load(fn, agg) for lab, fn, col in FILES]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.70, bottom=0.15, wspace=0.3)

    # -- A: margin, LHS - RHS. above zero = CCE priority is the better bet ----
    rng = np.random.default_rng(0)
    for i, (lab, col, lhs, rhs) in enumerate(data):
        m = lhs - rhs
        x = i + (rng.random(len(m)) - 0.5) * 0.20
        axa.scatter(x, m, s=95, color=col, alpha=0.9, lw=1.6, edgecolor="white", zorder=3)
        axa.plot([i - 0.26, i + 0.26], [np.median(m)] * 2, lw=2.6, color=col, zorder=4)
        wins = int((m >= 0).sum())
        axa.text(i, 1.02, f"CCE wins {wins}/{len(m)}", ha="center", va="bottom",
                 fontsize=11.5, color=col, fontweight="bold",
                 transform=axa.get_xaxis_transform())
    axa.axhline(0, color=INK2, lw=1.5, ls=(0, (5, 3)), zorder=2)
    axa.set_yscale("symlog", linthresh=1e-4)
    axa.set_xticks(range(len(data)))
    axa.set_xticklabels([d[0] for d in data], fontsize=10.5, color=INK2)
    axa.set_xlim(-0.6, len(data) - 0.4)
    axa.set_ylabel("Cov(c,u)/E[c]  −  Cov(d,u)/E[d]", color=INK2, fontsize=11)
    axa.set_title("Above the line, CCE is the better signal", color=INK,
                  fontsize=12.5, fontweight="bold", loc="left", pad=28)
    axa.text(0.015, 0.03, "each dot = one converged run · bar = median",
             transform=axa.transAxes, fontsize=9.5, color=INK2, va="bottom")

    # -- B: both sides flip --------------------------------------------------
    w = 0.34
    for i, (lab, col, lhs, rhs) in enumerate(data):
        axb.bar(i - w / 2, np.median(lhs), w, color=col, zorder=3,
                label="CCE side  Cov(c,u)/E[c]" if i == 0 else None)
        axb.bar(i + w / 2, np.median(rhs), w, color=col, alpha=0.42, zorder=3,
                hatch="///", edgecolor=col, lw=0,
                label="TD side  Cov(d,u)/E[d]" if i == 0 else None)
    axb.axhline(0, color=INK2, lw=1.5, zorder=4)
    axb.set_yscale("symlog", linthresh=1e-4)
    axb.set_xticks(range(len(data)))
    axb.set_xticklabels([d[0] for d in data], fontsize=10.5, color=INK2)
    axb.set_xlim(-0.6, len(data) - 0.4)
    axb.set_ylabel("median covariance ratio", color=INK2, fontsize=11)
    axb.set_title("Both sides swap sign between environments", color=INK,
                  fontsize=12.5, fontweight="bold", loc="left", pad=10)
    axb.legend(frameon=False, fontsize=10, loc="lower left", labelcolor=INK2)

    for a in (axa, axb):
        a.grid(axis="y", color=GRID, lw=0.8, alpha=0.7, zorder=0)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9.5)

    # Computed, never hardcoded: a stale literal in the subtitle would silently
    # misreport the statistic if the underlying records changed.
    from scipy.stats import fisher_exact
    (_, _, l0, r0), (_, _, l1, r1) = data
    w0, w1 = int((l0 >= r0).sum()), int((l1 >= r1).sum())
    pval = fisher_exact([[w0, len(l0) - w0], [w1, len(l1) - w1]],
                        alternative="greater")[1]

    fig.suptitle("CCE's replay signal is useful only where CCE actually won",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.085, ha="left", y=0.945)
    fig.text(0.085, 0.845,
             "FrozenLake 8×8 · neutral DQN-uniform checkpoints · matched by achieved win rate · "
             f"{agg} aggregation · Fisher one-sided p = {pval:.3f}",
             fontsize=10, color=INK2, ha="left")

    path = os.path.join(OUT_DIR, "fig_predicate_by_env.png")
    fig.savefig(path, dpi=170, facecolor="white")
    print("wrote", path)
    for lab, col, lhs, rhs in data:
        print(f"  {lab.splitlines()[0]:>16}: n={len(lhs)}  "
              f"LHS med {np.median(lhs):+.2e}  RHS med {np.median(rhs):+.2e}  "
              f"CCE wins {int((lhs>=rhs).sum())}/{len(lhs)}")


if __name__ == "__main__":
    main()
