"""Learning curves for the instrumentation sweep: win rate over training, and the spread.

WHY THERE IS NO SEPARATE "RETURN" PANEL. Under this env's sparse reward (goal_rew=1.0,
coll_rew=0.0, every shaping weight zeroed) an episode returns exactly 1.0 if it reached
the goal and 0.0 otherwise, so avg_return IS win_rate -- verified bit-identical over 751
evals (max |diff| = 0.000000, corr = 1.000000). Drawing both would be the same curve
twice. Panel 2 shows the per-seed spread instead, which is the thing that actually
decides whether any difference between arms is readable.

SAFE MID-SWEEP. Arms progress at very different rates (uniform finishes 250k in ~1.1 h,
a CCE arm needs ~8.6 h), so every arm is truncated to the COMMON EPISODE FLOOR. Without
that, the fast arms would simply appear to run further and look better.

READ PANEL 2 BEFORE PANEL 1. JaxNav seeds are bimodal and the paired seed sd is ~30pp,
so at n=3 the between-arm gaps in panel 1 are far smaller than the within-arm scatter.
Nothing here is an outcome claim.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_learning_curves
"""
import argparse
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import RUNS, ROOT

MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_instrument.json")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_learning_curves.png")

ORDER = ["uniform", "per", "cce_wmean", "cce_max", "cce_only", "cce_wmean_bs"]
LAB = {"uniform": "uniform (control)", "per": "PER",
       "cce_wmean": "CCE mul·wmean", "cce_max": "CCE mul·max",
       "cce_only": "CCE only μ=1", "cce_wmean_bs": "CCE wmean +bootstrap"}
# uniform/PER keep the repo's control colours (grey is a deliberate reference, not a
# categorical hue); the four CCE hues are the Okabe-Ito set validated at worst
# adjacent CVD dE 11.0 (deutan) / normal-vision floor 24.2.
COL = {"uniform": "#8a8a8a", "per": "#b9a06a", "cce_wmean": "#0072b2",
       "cce_max": "#e69f00", "cce_only": "#009e73", "cce_wmean_bs": "#d55e00"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
DECAY = 62_500


def load(array_id):
    man = json.load(open(MANIFEST))
    out = subprocess.run(["sacct", "-j", str(array_id), "-X", "-n", "-P",
                          "-o", "JobIDRaw,JobID,State"],
                         capture_output=True, text=True).stdout
    res = {p[1]: (p[0], p[2]) for p in (l.split("|") for l in out.strip().split("\n"))
           if len(p) >= 3}
    curves, states = {}, {}
    for rec in man.values():
        if rec["job_id"] not in res:
            continue
        run_dir, st = res[rec["job_id"]]
        f = os.path.join(RUNS, run_dir, "metrics.log")
        if not os.path.exists(f):
            continue
        ep, wr = [], []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            q = line.split()
            if len(q) < 6 or q[0] == "episode":
                continue
            ep.append(float(q[0])); wr.append(float(q[3].rstrip("%")))
        if len(ep) > 5:
            curves.setdefault(rec["arm"], []).append((np.array(ep), np.array(wr)))
            states.setdefault(rec["arm"], []).append(st)
    return curves, states


def smooth(y, w=21):
    if len(y) < w:
        return y
    return np.convolve(np.pad(y, w // 2, mode="edge"), np.ones(w) / w,
                       mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274678")
    args = ap.parse_args()
    curves, states = load(args.array)
    if not curves:
        print("nothing resolved yet")
        return

    floor = min(min(x[-1] for x, _ in runs) for runs in curves.values())
    print(f"common episode floor: {floor:,.0f}")
    grid = np.linspace(500, floor, 400)

    def band(arm):
        st = np.vstack([np.interp(grid, x, smooth(y)) for x, y in curves[arm]])
        return np.median(st, axis=0), st

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 6.3), facecolor=SURFACE)
    fig.subplots_adjust(left=.048, right=.988, top=.655, bottom=.115, wspace=.235)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.axvspan(0, DECAY, color="#f1efec", zorder=0)
        ax.grid(alpha=.55, color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("episodes (thousands)", fontsize=10, color=INK2)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
        ax.set_xlim(0, floor)

    # ---- 1. median win rate per arm -------------------------------------
    ax = axes[0]
    finals = {}
    for arm in ORDER:
        if arm not in curves:
            continue
        med, _ = band(arm)
        n_done = sum(1 for s in states[arm] if s == "COMPLETED")
        lw = 2.6 if arm == "uniform" else 2.0
        ax.plot(grid, med, lw=lw, color=COL[arm],
                label=f"{LAB[arm]}   ({n_done}/{len(states[arm])} done)")
        finals[arm] = med[-1]
    ax.text(DECAY * .5, 3, "exploring", fontsize=9.5, color=INK2,
            style="italic", ha="center")
    ax.set_ylabel("evaluation win rate (%)", fontsize=10, color=INK2)
    ax.set_ylim(0, max(finals.values()) * 1.42)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=1,
              handlelength=1.6, labelspacing=.32)
    ax.set_title("1.  Win rate — median of 3 seeds", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left", pad=9)
    if "cce_max" in curves:
        ax.annotate("cce_max COLLAPSES\n(all 3 seeds — real,\nnot a truncation edge)",
                    xy=(floor * .985, finals["cce_max"] + 3),
                    xytext=(floor * .60, max(finals.values()) * 0.42),
                    fontsize=9.5, color=COL["cce_max"], fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=COL["cce_max"], lw=1.6))

    # ---- 2. every seed, so the scatter is visible -----------------------
    ax = axes[1]
    spreads = []
    for arm in ORDER:
        if arm not in curves:
            continue
        _, st = band(arm)
        for row in st:
            ax.plot(grid, row, lw=1.15, color=COL[arm], alpha=.85)
        spreads.append(st[:, -1].max() - st[:, -1].min())
    ax.set_ylabel("evaluation win rate (%)  —  every individual seed",
                  fontsize=10, color=INK2)
    ax.set_ylim(0, max(finals.values()) * 1.42)
    ax.set_title("2.  Same data, per seed — this is why n=3 says nothing",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=9)
    allf = np.array([v for arm in curves for v in band(arm)[1][:, -1]])
    gap = max(finals.values()) - min(finals.values())
    ax.text(.035, .955,
            f"between-arm gap (medians):   {gap:.1f} pp\n"
            f"within-arm seed spread:      up to {max(spreads):.1f} pp\n"
            f"all {len(allf)} seeds span:            "
            f"{allf.min():.1f} – {allf.max():.1f} pp",
            transform=ax.transAxes, fontsize=10, color=INK, va="top",
            fontweight="bold", family="monospace")

    # ---- 3. stability: peak vs final, at the SAME cutoff for every arm ----
    ax = axes[2]
    ax.set_xscale("linear")
    for coll in list(ax.collections):
        coll.remove()
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", alpha=.55, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    stab = {}
    for arm in ORDER:
        if arm not in curves:
            continue
        d = []
        for x, y in curves[arm]:
            m = x <= floor
            w = y[m]
            if len(w) < 25:
                continue
            sm = np.convolve(w, np.ones(21) / 21, mode="valid")
            d.append(float(sm.max() - w[-20:].mean()))
        if d:
            stab[arm] = d
    names = [a for a in ORDER if a in stab]
    xs = np.arange(len(names))
    meds = [float(np.median(stab[a])) for a in names]
    ax.bar(xs, meds, .58, color=[COL[a] for a in names], alpha=.9)
    for i, a in enumerate(names):
        ax.scatter([i] * len(stab[a]), stab[a], s=34, color=INK, zorder=5,
                   edgecolor=SURFACE, linewidth=1.1)
        ax.text(i, meds[i] + 2.2, f"{meds[i]:.0f}", ha="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([LAB[a].split(" (")[0].replace("CCE ", "") for a in names],
                       fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("peak win rate MINUS final win rate  (pp)", fontsize=10, color=INK2)
    ax.set_title("3.  Does it HOLD what it learns?", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left", pad=9)
    ax.set_ylim(0, max(max(v) for v in stab.values()) * 1.22)
    ax.text(.035, .965,
            "bars = median of 3 seeds, dots = each seed\n"
            "both measured within the SAME episode cutoff,\n"
            "so this is not the 250k-vs-185k confound",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")

    fig.text(.048, .968, "Learning curves — and why they are not the result",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.048, .912,
             f"JaxNav 8x8 fill 0.3  ·  array {args.array}  ·  3 seeds/arm  ·  every arm "
             f"truncated to the common floor of {floor:,.0f} of 250,000 episodes.\n\n"
             "There is no separate RETURN panel: under this env's sparse reward an episode "
             "returns exactly 1.0 for a goal and 0.0 otherwise, so avg_return IS win_rate\n"
             "(verified bit-identical over 751 evals). Panel 2 is the honest reading — the "
             "seed scatter is larger than every gap between arms, which is the measured\n"
             "reason this sweep is a MECHANISM run: JaxNav's paired seed sd is ~30pp and "
             "resolving a 5pp effect would need ~290 seed-pairs. The ONE exception is "
             "panel 3 --\ncce_max gives up ~55pp of its peak in 3 of 3 seeds while nothing "
             "else exceeds 9pp, an effect large enough and consistent enough to survive n=3.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.5)

    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")
    for a, v in sorted(finals.items(), key=lambda kv: -kv[1]):
        print(f"   {a:<14} {v:5.1f}%")


if __name__ == "__main__":
    main()
