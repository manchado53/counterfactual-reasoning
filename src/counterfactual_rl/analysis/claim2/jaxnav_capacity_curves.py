"""JaxNav training curves for the buffer-capacity experiment (array 275241).

Answers what the endpoint numbers cannot: do the small-buffer arms fail to learn at all,
or do they learn and then fall over? 8 seeds/arm, full 250k episodes.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_capacity_curves
"""
import json, os, subprocess, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .jaxnav_bootstrap_curves import ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_capacity_curves.png")
R = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
MAN = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_capacity_outcome.json")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
COL = {'uniform': "#8a8a8a", 'cce_base': "#0072b2",
       'cce_8192': "#e69f00", 'cce_2500': "#d55e00"}
LAB = {'uniform': "uniform  (100k buffer)", 'cce_base': "CCE  100k   cov 0.86%",
       'cce_8192': "CCE  8,192  cov 1.6%", 'cce_2500': "CCE  2,500  cov 5.1%"}
ORDER = ['uniform', 'cce_base', 'cce_8192', 'cce_2500']
DECAY = 62_500


def load():
    m = json.load(open(MAN)); aid = sorted({v['array_id'] for v in m.values()})[0]
    out = subprocess.run(["sacct", "-j", aid, "-X", "-n", "-P", "-o", "JobIDRaw,JobID,State"],
                         capture_output=True, text=True).stdout
    res = {p[1]: (p[0], p[2]) for p in (l.split("|") for l in out.strip().split("\n")) if len(p) >= 3}
    per = collections.defaultdict(list)
    for k, v in m.items():
        if v['job_id'] not in res: continue
        rd, s = res[v['job_id']]
        if s != 'COMPLETED': continue
        f = os.path.join(R, rd, 'metrics.log')
        if not os.path.exists(f): continue
        ep, wr = [], []
        for line in open(f):
            if line.startswith('#') or not line.strip(): continue
            q = line.split()
            if len(q) >= 6 and q[0] != 'episode':
                ep.append(float(q[0])); wr.append(float(q[3].rstrip('%')))
        if len(ep) > 20: per[v['arm']].append((np.array(ep), np.array(wr)))
    return per


def smooth(y, w=21):
    if len(y) < w: return y
    return np.convolve(np.pad(y, w//2, mode="edge"), np.ones(w)/w, mode="valid")[:len(y)]


def main():
    per = load()
    if not per: print("nothing"); return
    floor = min(min(x[-1] for x, _ in v) for v in per.values())
    grid = np.linspace(500, floor, 400)
    print(f"common floor {floor:,.0f} episodes; runs/arm " +
          str({a: len(v) for a, v in per.items()}))

    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.4), facecolor=SURFACE)
    fig.subplots_adjust(left=.06, right=.985, top=.70, bottom=.12, wspace=.19)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.axvspan(0, DECAY, color="#f1efec", zorder=0)
        ax.grid(alpha=.55, color=GRID, lw=.8); ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_xlabel("episodes (thousands)", fontsize=10.5, color=INK2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v/1000:.0f}"))
        ax.set_xlim(0, floor); ax.set_ylim(0, 80)

    ax = axes[0]
    for a in ORDER:
        if a not in per: continue
        st = np.vstack([np.interp(grid, x, smooth(y)) for x, y in per[a]])
        ax.plot(grid, np.median(st, axis=0), lw=2.6, color=COL[a],
                label=f"{LAB[a]}   (n={len(per[a])})")
    ax.text(DECAY*.5, 2.5, "exploring", fontsize=9.5, color=INK2, style="italic", ha="center")
    ax.set_ylabel("evaluation win rate (%)", fontsize=10.5, color=INK2)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.set_title("1.  Median of 8 seeds", fontsize=12.5, fontweight="bold",
                 color=INK, loc="left", pad=9)

    ax = axes[1]
    for a in ORDER:
        if a not in per: continue
        for x, y in per[a]:
            ax.plot(grid, np.interp(grid, x, smooth(y)), lw=1.15, color=COL[a], alpha=.8)
    ax.axhline(40, color=INK2, ls="--", lw=1.4)
    ax.text(floor*.98, 42, "good mode", ha="right", fontsize=9.5, color=INK2)
    ax.set_ylabel("evaluation win rate (%) — every seed", fontsize=10.5, color=INK2)
    ax.set_title("2.  Every individual seed", fontsize=12.5, fontweight="bold",
                 color=INK, loc="left", pad=9)
    ax.text(.97, .95, "8,192 rises to ~25% by 90k then DECAYS\n"
                      "2,500 never rises at all",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            color=COL['cce_8192'], fontweight="bold")

    fig.text(.06, .955, "JaxNav: shrinking the buffer to raise coverage",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.06, .885,
             "8x8 fill 0.3 · array 275241 · 8 seeds/arm · full 250,000 episodes · "
             "grey band = epsilon still decaying.\n\n"
             "Coverage rises 0.86% -> 1.6% -> 5.1% across the three CCE arms. Both reduced-"
             "capacity arms end at 0/8 seeds in the good mode (p=0.001).\n"
             "cce_8192 holds exactly one collection chunk and discards NO data, so the "
             "failure is lost DIVERSITY -- the buffer is nearly on-policy -- not lost samples.\n"
             "Note the two failure SHAPES differ: 8,192 learns to ~25% by 90k and then decays, "
             "while 2,500 never leaves the floor. And uniform tracks CCE-100k almost exactly "
             "until ~90k,\nwhich is the JaxNav null itself: at the shipped capacity, CCE is "
             "indistinguishable from plain replay.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.5)
    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
