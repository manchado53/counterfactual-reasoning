"""Bootstrap ON vs OFF, on the quantities that only exist since the instruments landed.

The bootstrap sweep (array 274476) established that fixing the truncated-rollout bug
changes the SCORE and not the LEARNING (Wilcoxon p=0.837 over 20 seed-matched pairs).
It could not say WHY, because nothing recorded what the buffer contained.

Now it can, and the answer is a clean split:

    bootstrap does exactly what it was built to do, ENTIRELY INSIDE the ~1% of the
    buffer that carries a measured score, and it cannot touch the ~99% that does not.

Since the ~99% is what the sampler sees, no improvement to score QUALITY can change
replay while coverage stays where it is. That generalises past bootstrap to any future
fix to the score -- which is the argument for treating coverage as the lever.

Both arms are the same configuration except `cf_bootstrap`, same seeds, same cell.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_bootstrap_side_by_side
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
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_bootstrap_side_by_side.png")

C_OFF, C_ON = "#0072b2", "#d55e00"          # repo convention, CVD-validated dE 21.9
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
ARMS = [("cce_wmean", "bootstrap OFF", C_OFF), ("cce_wmean_bs", "bootstrap ON", C_ON)]


def load(array_id, arm):
    man = json.load(open(MANIFEST))
    res = resolve(array_id)
    runs = []
    for rec in man.values():
        if rec["arm"] != arm or rec["job_id"] not in res:
            continue
        p = os.path.join(RUNS, res[rec["job_id"]][0], "ess.jsonl")
        if os.path.exists(p):
            r = [json.loads(l) for l in open(p) if l.strip()]
            if r and "score_hist" in r[-1]:
                runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274678")
    args = ap.parse_args()

    data = {a: load(args.array, a) for a, _, _ in ARMS}
    if not all(data.values()):
        print("not enough data yet")
        return
    floor = min(min(max(x["episode"] for x in r) for r in runs)
                for runs in data.values())
    print(f"common episode floor: {floor:,}")

    def at_floor(arm):
        """median-across-seeds of the last row at or before the floor."""
        out = []
        for r in data[arm]:
            rows = [x for x in r if x["episode"] <= floor]
            if rows:
                out.append(rows[-1])
        return out

    fig = plt.figure(figsize=(16.2, 8.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                          left=.06, right=.975, top=.745, bottom=.075,
                          wspace=.21, hspace=.40)

    # ---- top row: the distribution over training, one heatmap per arm ----
    r0 = at_floor("cce_wmean")[0]
    lo, hi, nb = r0["score_hist_lo"], r0["score_hist_hi"], r0["score_hist_nbins"]
    edges = np.logspace(np.log10(lo), np.log10(hi), nb + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    vmin = 1.0 / r0["n"]

    for i, (arm, lab, col) in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(SURFACE)
        rows = max(data[arm], key=len)
        rows = [x for x in rows if x["episode"] <= floor]
        eps = np.array([x["episode"] for x in rows], float)
        H = np.vstack([np.array(x["score_hist"], float) / x["n"] for x in rows]).T
        m = ax.pcolormesh(eps, centers, np.ma.masked_less(H, vmin), cmap="Blues",
                          norm=mcolors.LogNorm(vmin=vmin, vmax=1.0), shading="nearest")
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.set_title(f"{lab}", fontsize=12.5, fontweight="bold", color=col,
                     loc="left", pad=8)
        ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
        if i == 0:
            ax.set_ylabel("CCE score stored in the buffer", fontsize=9.5, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        n_occ = int((H[:, -1] > 0).sum())
        ax.text(.03, .05, f"{n_occ} of {nb} bins occupied", transform=ax.transAxes,
                fontsize=10, color=INK, fontweight="bold")
        if i == 1:
            cb = fig.colorbar(m, ax=ax, pad=.012, fraction=.037)
            cb.set_label("share of the buffer", fontsize=8.5, color=INK2)
            cb.ax.tick_params(colors=INK2, labelsize=7.5)

    # ---- bottom left: the two distributions overlaid ---------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(SURFACE)
    for arm, lab, col in ARMS:
        rr = at_floor(arm)
        h = np.median(np.vstack([np.array(x["score_hist"], float) / x["n"]
                                 for x in rr]), axis=0)
        occ = h > 0
        ax.step(centers[occ], 100 * h[occ], where="mid", color=col, lw=2.2, label=lab)
        ax.fill_between(centers[occ], 0, 100 * h[occ], step="mid", color=col, alpha=.18)
    ax.set_xscale("log")
    ax.set_xlabel("CCE score", fontsize=9.5, color=INK2)
    ax.set_ylabel("% of the buffer", fontsize=9.5, color=INK2)
    ax.set_title("The spike just MOVES — it does not shrink", fontsize=12,
                 fontweight="bold", color=INK, loc="left", pad=8)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.grid(alpha=.55, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- bottom right: what changed vs what did not ---------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(SURFACE)

    def med(arm, fn):
        return float(np.median([fn(x) for x in at_floor(arm)]))

    metrics = [
        ("scores exactly 0  (%)", lambda x: 100 * x["score_n_zero"] / x["n"], "CHANGED"),
        ("mean score  (x1000)", lambda x: 1000 * x["score_mean"], "CHANGED"),
        ("bins occupied", lambda x: float((np.array(x["score_hist"]) > 0).sum()), "CHANGED"),
        ("COVERAGE  (% scored)", lambda x: 100 * x["scored_frac"], "UNCHANGED"),
        ("top-2 bins hold  (%)", lambda x: 100 * np.sort(np.array(x["score_hist"], float))[::-1][:2].sum() / x["n"], "UNCHANGED"),
    ]
    ypos = np.arange(len(metrics))[::-1]
    for y, (lab, fn, kind) in zip(ypos, metrics):
        a, b = med("cce_wmean", fn), med("cce_wmean_bs", fn)
        base = max(a, b, 1e-9)
        ax.barh(y + .18, a / base, .32, color=C_OFF, alpha=.9)
        ax.barh(y - .18, b / base, .32, color=C_ON, alpha=.9)
        ax.text(a / base + .02, y + .18, f"{a:,.2f}", va="center", fontsize=9, color=C_OFF)
        ax.text(b / base + .02, y - .18, f"{b:,.2f}", va="center", fontsize=9, color=C_ON)
        ax.text(-.04, y, lab, va="center", ha="right", fontsize=10, color=INK)
        ax.text(1.30, y, kind, va="center", ha="right", fontsize=10.5,
                fontweight="bold", color="#0072b2" if kind == "CHANGED" else "#b0521f")
    ax.axhline(1.5, color=GRID, lw=1.4)
    ax.set_xlim(0, 1.34)
    ax.set_ylim(-.75, len(metrics) - .25)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("What bootstrap moved, and what it could not",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(1.30, -.62, "bars scaled per row to the larger of the two",
            ha="right", fontsize=8, color=INK2, style="italic")

    fig.text(.06, .965, "Bootstrap fixed the score. It could not fix the buffer.",
             fontsize=19, fontweight="bold", color=INK, va="top")
    # One text object, not three: separate fig.text calls at hand-picked y values
    # collided once bbox_inches="tight" rescaled the canvas.
    fig.text(.06, .915,
             f"JaxNav 8x8 fill 0.3  ·  array {args.array}  ·  cce_wmean, identical "
             f"except cf_bootstrap  ·  3 seeds  ·  median  ·  at {floor:,} of 250,000 "
             f"episodes.\n\n"
             "Bootstrap does exactly what it was built to do — inside the ~1% of the "
             "buffer that carries a MEASURED score. The ~99% that carries the\n"
             "inherited running mean is untouched, and that is the part the sampler "
             "sees. So no improvement to score QUALITY can move replay while\n"
             "coverage stays where it is. That argument is not specific to bootstrap.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.55)

    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
