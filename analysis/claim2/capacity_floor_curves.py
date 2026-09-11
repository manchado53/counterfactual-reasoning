"""Training curves for the capacity ladder (array 275562) — uniform vs PER, six capacities.

The final-value figure (capacity_floor.py) says WHERE the collapse lands. This says WHEN --
and the answer corrects the obvious guess. The small buffers do NOT fail to learn. They
learn, peak around episode 65k-125k, and then LOSE it:

    uniform   8,192   peak 26.5% @ 67,500   ->  final  7.3%   -19.3pp
    uniform  12,288   peak 36.3% @ 125,000  ->  final 15.0%   -21.3pp
    uniform  16,384   peak 37.1% @ 120,000  ->  final 18.4%   -18.7pp
    uniform  25,000   peak 48.3% @ 247,500  ->  final 46.0%    -2.4pp
    uniform  50,000   peak 60.4% @ 250,000  ->  final 60.4%    +0.0pp

That is catastrophic forgetting, which is exactly what too little replay diversity predicts
-- and it is the mechanism, not just the outcome. 25,000 and 50,000 are still RISING at
250k; every rung below 25,000 has already peaked and fallen.

Capacity is an ORDERED variable, so it gets one sequential colour ramp shared by both
panels -- the same colour means the same capacity on the left and on the right. Lines are
per-capacity medians over 5 seeds (26 / 18 at the pooled default), with the IQR band drawn
only for the two extremes so the panel stays readable.

Everything left of the epsilon marker is exploration, not an algorithm comparison.
"""
import os

import numpy as np

import greedy_learning_curves as G

OUT = os.path.join(G.WT, "docs/figures/real/claim2/jaxnav")
CAPS = [8_192, 12_288, 16_384, 25_000, 50_000, 100_000]
BURST = 8_192
PANELS = [("uniform", "uniform  —  no priorities at all"),
          ("per",     "PER  —  TD-error priorities")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = G.collect(G._label_floor)
    cmap = plt.get_cmap("viridis")
    # position on the ramp by log capacity, so the spacing matches the x-axis of the
    # final-value figure rather than bunching the three small rungs together
    lo, hi = np.log(CAPS[0]), np.log(CAPS[-1])
    colour = {cap: cmap(0.08 + 0.84 * (np.log(cap) - lo) / (hi - lo)) for cap in CAPS}

    fig = plt.figure(figsize=(15, 6.4), dpi=150)
    axes = [fig.add_axes([.055, .135, .425, .70]), fig.add_axes([.525, .135, .425, .70])]

    for ax, (key, title) in zip(axes, PANELS):
        for cap in CAPS:
            v = c[f"{key}_{cap}"]
            if cap in (CAPS[0], CAPS[-1]):
                ax.fill_between(G.GRID, v["lo"], v["hi"], color=colour[cap], alpha=.13, lw=0)
            ax.plot(G.GRID, v["med"], color=colour[cap], lw=2.4,
                    label=f"{cap:,}" + ("   (default)" if cap == 100_000 else
                                        "   (1 burst)" if cap == BURST else ""))
        ax.axvline(G.EPS_DECAY, color="#a8341c", lw=1.1, ls="--")
        ax.set_xlim(0, 250_000); ax.set_ylim(0, 80)
        ax.set_xlabel("episode", fontsize=10.5)
        ax.grid(alpha=.18, lw=.7)
        ax.set_title(title, fontsize=11.5, loc="left", pad=8)
        ax.legend(fontsize=8.5, loc="upper left", frameon=False, title="buffer capacity",
                  title_fontsize=8.5, ncol=2, columnspacing=1.1, handlelength=1.6)
    axes[0].set_ylabel("win rate  (%)", fontsize=10.5)
    axes[0].text(G.EPS_DECAY + 4000, 3.0, "epsilon decay ends\nleft of here = exploration",
                 fontsize=8, color="#a8341c", va="bottom")
    axes[1].set_yticklabels([])

    fig.text(.055, .955,
             "Capacity ladder, training curves   (JaxNav 8x8, fill 0.3, 5 seeds per rung, "
             "NO consequence scoring in any of these 60 runs)",
             fontsize=12.5, weight="bold")
    fig.text(.055, .905,
             "The small buffers DO learn — then they forget. Every rung below 25,000 peaks "
             "at 26-37% around episode 65k-125k and loses ~19pp by the end; 25,000 and 50,000 "
             "are still rising at 250k.\nThat is catastrophic forgetting, which is what too "
             "little replay diversity predicts. Both panels do it, so the floor is a "
             "replay-diversity effect, not a prioritisation effect.",
             fontsize=9.5, color="#33444d")

    p = os.path.join(OUT, "capacity_floor_curves.png")
    fig.savefig(p, facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
