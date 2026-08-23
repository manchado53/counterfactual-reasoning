"""Old 500k sweep vs the new factorial, same environment, overlaid.

Both are 8x8 fill=0.1 holes maps with 25 and 5 seeds respectively. They differ
in two ways that matter:

  budget    500k vs 250k
  scoring   20 rollouts / horizon 20   vs   40 rollouts / horizon 60

The point of the figure is the vertical line at ep 250k. Left of it the two
sweeps agree and every arm is tied. Right of it -- only reachable in the old
sweep -- uniform loses ~56 points and PER ~8, while the CCE arms hold. That
collapse window is where P(CCE+max > PER) = 0.812 came from, so the prior
"CCE beats PER" result is really "CCE degrades less", and the new sweep stops
before it can be observed.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compute_metrics import iqm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data")
RUNS = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_sweep_compare.png")

OLD_ARMS = {"cce_max": range(272275, 272300), "cce_wmean": range(272300, 272325),
            "per": range(272325, 272350), "uniform": range(272350, 272375)}
COL = {"uniform": "#6e6e6e", "per": "#0072b2", "cce_wmean": "#009e73", "cce_max": "#d55e00"}
LAB = {"uniform": "uniform", "per": "PER", "cce_wmean": "CCE mul·wmean", "cce_max": "CCE mul·max"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def smooth(y, w=25):
    if len(y) < w:
        return y
    pad = w // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(w) / w, mode="valid")[:len(y)]


def old_curves():
    z = np.load(os.path.join(DATA, "curves_25seed_500k.npz"))
    out = {}
    for arm, jobs in OLD_ARMS.items():
        c = [(z[f"{j}_ep"], z[f"{j}_win"]) for j in jobs if f"{j}_ep" in z]
        if c:
            out[arm] = c
    return out


def new_curves():
    man = json.load(open(os.path.join(DATA, "manifest_factorial.json")))
    out = {}
    for rec in man.values():
        if rec["cell"] != "8x8_f01" or rec["arm"] not in OLD_ARMS:
            continue
        rd = rec.get("run_dir")
        f = os.path.join(RUNS, str(rd), "metrics.log") if rd else None
        if not f or not os.path.exists(f):
            continue
        ep, wr = [], []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 5 or p[0] == "episode":
                continue
            ep.append(float(p[0])); wr.append(float(p[3].rstrip("%")) / 100)
        if len(ep) > 5:
            out.setdefault(rec["arm"], []).append((np.array(ep), np.array(wr)))
    return out


def draw(ax, curves, title, sub):
    for arm, runs in curves.items():
        hi = min(e[-1] for e, _ in runs)
        grid = np.linspace(250, hi, 300)
        stack = np.vstack([np.interp(grid, e, w) for e, w in runs])
        y = smooth(np.array([iqm(c) for c in stack.T]))
        ax.plot(grid / 1000, y * 100, color=COL[arm], lw=2.2,
                label=f"{LAB[arm]} (n={len(runs)})")
    ax.axvline(250, color=INK2, ls="--", lw=1.4)
    ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_title(f"{title}\n{sub}", fontsize=11.5, fontweight="bold", color=INK,
                 loc="left", pad=8, linespacing=1.9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.8))
    fig.subplots_adjust(left=.06, right=.985, top=.72, bottom=.11, wspace=.16)

    old, new = old_curves(), new_curves()
    draw(ax1, old, "OLD sweep — 500k budget",
         "20 rollouts, horizon 20 · 25 seeds/arm")
    ax1.set_xlim(0, 500); ax1.set_ylabel("evaluation win rate (%)", fontsize=9.5, color=INK2)
    ax1.axvspan(250, 500, color="#d55e00", alpha=.06, zorder=0)
    ax1.text(375, 92, "the collapse window", ha="center", fontsize=9,
             color="#a03c10", style="italic")
    draw(ax2, new, "NEW factorial — 250k budget",
         "40 rollouts, horizon 60 · 5 seeds/arm")
    ax2.set_xlim(0, 500)
    ax2.text(375, 50, "not run", ha="center", fontsize=11, color="#b6b1a9", style="italic")

    fig.text(.06, .945, "Same environment (8×8, fill 0.1) — why the old sweep showed a CCE win",
             fontsize=15.5, fontweight="bold", color=INK)
    fig.text(.06, .898,
             "Dashed line = episode 250k, where the new sweep stops. LEFT of it the two sweeps "
             "agree and every arm is tied.",
             fontsize=9.5, color=INK2)
    fig.text(.06, .864,
             "RIGHT of it, only the old sweep runs: uniform loses ~56 points and PER ~8, while "
             "the CCE arms hold. That is where P(CCE+max > PER) = 0.812 came from.",
             fontsize=9.5, color=INK2)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
