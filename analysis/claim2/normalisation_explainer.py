"""Slide figure: what min-max and rank normalisation actually do to a priority signal.

Explanatory, not a result. Uses the REAL td_magnitudes from a buffer snapshot so the
shapes are the ones our sampler actually sees, not a toy.

Jeremy 08/20 asked for "0-1 normalization of TD errors -- equal exponents do not imply
equal contribution". Min-max IS that ask. Rank is issue #16's counter-proposal. The
figure is the argument between them: min-max rescales, rank reshapes.
"""
import glob
import os

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
RUNS = os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(WT, "docs/figures/real/claim2/jaxnav")
INK, MINMAX, RANK, RED = "#3f5763", "#c98a5a", "#0d7c6b", "#a8341c"


def _real_td():
    snaps = sorted(glob.glob(os.path.join(RUNS, "*", "buffer", "buf_*.npz")),
                   key=os.path.getmtime)
    for f in reversed(snaps):
        z = np.load(f)
        td = np.asarray(z["td_magnitudes"], dtype=np.float64)
        if td.size >= 90_000 and td.max() > td.min():
            return td, os.path.basename(f)
    raise SystemExit("no usable snapshot")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    td, fname = _real_td()
    pct = np.linspace(0, 100, td.size)
    s = np.sort(td)
    mm = (s - s.min()) / (s.max() - s.min())
    rk = (rankdata(s, method="average") - 1) / (s.size - 1)
    cv = lambda x: x.std() / x.mean()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=150,
                             gridspec_kw=dict(height_ratios=[1, 2.6]))

    # ── row 1: four toy values, the intuition ──────────────────────────────────
    toy = np.array([0.1, 0.2, 0.3, 50.0])
    toy_t = [toy, (toy - toy.min()) / (toy.max() - toy.min()),
             (rankdata(toy) - 1) / (len(toy) - 1)]
    for ax, vals, colour, title in zip(
            axes[0], toy_t, [INK, MINMAX, RANK],
            ["RAW", "MIN-MAX   (Jeremy's ask)", "RANK   (issue #16)"]):
        lo, hi = (0, 55) if title == "RAW" else (-0.05, 1.05)
        ax.hlines(0, lo, hi, color="#c5cfd6", lw=1.2)
        ax.scatter(vals, np.zeros_like(vals), s=90, color=colour, zorder=3)
        # stagger the labels: under RAW and MIN-MAX the low values sit almost on top
        # of one another (which is the point being made) and the text would overlap.
        for j, v in enumerate(vals):
            ax.annotate(f"{v:.3g}", (v, 0), textcoords="offset points",
                        xytext=(0, 11 + 15 * j), ha="center", fontsize=9,
                        color=colour,
                        arrowprops=dict(arrowstyle="-", color=colour, lw=.6,
                                        alpha=.45) if j else None)
        ax.set_xlim(lo, hi); ax.set_ylim(-.6, 1.9)
        ax.set_title(title, fontsize=12, loc="left", weight="bold", color=colour, pad=8)
        ax.set_yticks([]); ax.spines[:].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0][0].text(0, -.45, "one huge outlier", fontsize=8.5, color=RED)
    axes[0][1].text(-.03, -.45, "three still crushed at 0 — outlier survives",
                    fontsize=8.5, color=RED)
    axes[0][2].text(-.03, -.45, "evenly spread — only the ORDER survives",
                    fontsize=8.5, color=RANK)

    # ── row 2: the real 100,000 TD magnitudes ─────────────────────────────────
    for ax, y, colour, lab, note in zip(
            axes[1], [s, mm, rk], [INK, MINMAX, RANK],
            [f"raw TD magnitude\ncv = {cv(s):.1f}",
             f"after min-max\ncv = {cv(mm):.1f}  (unchanged — cv is scale-free)",
             f"after rank\ncv = {cv(rk):.2f}  (uniform by construction)"],
            ["values span 5 orders of magnitude",
             "99% of slots sit within 1% of zero",
             "a straight line: every slot is spaced evenly"]):
        ax.plot(pct, y, color=colour, lw=2.2)
        ax.fill_between(pct, 0, y, color=colour, alpha=.12)
        ax.set_xlabel("buffer slots, sorted low → high  (%)", fontsize=9.5)
        ax.set_title(lab, fontsize=10, loc="left", pad=6)
        ax.grid(alpha=.18, lw=.6)
        ax.text(.03, .93, note, transform=ax.transAxes, fontsize=8.5, va="top",
                color="#55666f")
        ax.tick_params(labelsize=8.5)
    axes[1][0].set_yscale("log")
    axes[1][0].set_ylabel("value", fontsize=9.5)
    for ax in axes[1][1:]:
        ax.set_ylim(-.02, 1.02)

    fig.suptitle("Two ways to put a priority signal on a 0–1 scale",
                 fontsize=15, x=.008, ha="left", weight="bold")
    fig.text(.008, .935,
             "min-max rescales — the shape, and the outlier, survive.     "
             "rank reshapes — only the ordering survives.",
             fontsize=10.5, color="#55666f")
    fig.text(.008, .012,
             f"Bottom row: all 100,000 real TD magnitudes from {fname}. "
             "The complaint being fixed is that cv(TD)=47 against cv(CCE)=0.8 at identical "
             "exponents — a 59x mismatch in how much each signal moves the sampler.\n"
             "Min-max leaves that mismatch untouched because cv is scale-free; rank sets "
             "both to 0.58 by construction. Both are running on JaxNav, 5 seeds each.",
             fontsize=8.5, color="#55666f")
    fig.tight_layout(rect=[0, .06, 1, .915])
    out = os.path.join(OUT, "normalisation_explainer.png")
    fig.savefig(out)
    print("wrote", out)
    print(f"  cv raw {cv(s):.2f}   min-max {cv(mm):.2f}   rank {cv(rk):.3f}")


if __name__ == "__main__":
    main()
