"""Win-rate curves for the bootstrap sweep: cf_bootstrap OFF vs ON, paired.

One panel per CCE variant. Within a panel the ONLY difference between the two
coloured lines is `cf_bootstrap` -- same seeds, same cell, same everything else --
so the gap between them is the bootstrap effect and nothing else. uniform and PER
are drawn in every panel as the shared controls.

Safe to run mid-sweep. Run dirs are resolved from sacct (array tasks write to
JobIDRaw, not <array>_<task>), curves are drawn only as far as each arm's seeds
have actually reached, and an arm still running is drawn DASHED with its progress
in the legend, so a partially-trained arm can never be mistaken for a finished one.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_bootstrap_curves [--array 274476]
"""
import argparse
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compute_metrics import iqm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_bootstrap.json")
RUNS = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_bootstrap_curves.png")

TARGET, DECAY = 250_000, 62_500
VARIANTS = ["cce_wmean", "cce_max", "cce_add", "cce_only"]
LAB = {"cce_wmean": "CCE mul·wmean", "cce_max": "CCE mul·max",
       "cce_add": "CCE add μ=.25", "cce_only": "CCE only μ=1"}
C_OFF, C_ON = "#0072b2", "#d55e00"        # bootstrap off / on
C_UNI, C_PER = "#8a8a8a", "#b9a06a"       # controls
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def resolve(array_id):
    """job_id (<array>_<task>) -> JobIDRaw, which is the run-dir name."""
    out = subprocess.run(
        ["sacct", "-j", str(array_id), "-X", "-n", "-P", "-o", "JobIDRaw,JobID,State"],
        capture_output=True, text=True).stdout
    m = {}
    for line in out.strip().split("\n"):
        p = line.split("|")
        if len(p) >= 3:
            m[p[1]] = (p[0], p[2])
    return m


def load(array_id):
    """(arm) -> [(episodes, win_rate), ...] plus per-arm completion counts."""
    man = json.load(open(MANIFEST))
    resolved = resolve(array_id)
    curves, states = {}, {}
    for key, rec in man.items():
        jid = rec["job_id"]
        if jid not in resolved:
            continue
        run_dir, state = resolved[jid]
        f = os.path.join(RUNS, run_dir, "metrics.log")
        if not os.path.exists(f):
            continue
        ep, wr = [], []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 6 or p[0] == "episode":
                continue
            ep.append(float(p[0])); wr.append(float(p[3].rstrip("%")))
        if len(ep) > 3:
            curves.setdefault(rec["arm"], []).append((np.array(ep), np.array(wr)))
            states.setdefault(rec["arm"], []).append(state)
    return curves, states


def smooth(y, w=21):
    if len(y) < w:
        return y
    pad = w // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(w) / w, mode="valid")[:len(y)]


def draw(ax, runs, color, label, states):
    """IQM across seeds, extended only as far as at least half of them reached."""
    if not runs:
        return
    ends = sorted(e[-1] for e, _ in runs)
    hi = ends[len(ends) // 2]                     # median endpoint
    if hi < 2000:
        return
    grid = np.linspace(250, hi, 240)
    cols = []
    for g in grid:
        vals = [np.interp(g, e, w) for e, w in runs if e[-1] >= g]
        cols.append(iqm(np.array(vals)) if len(vals) >= 2 else np.nan)
    done = sum(1 for s in states if s == "COMPLETED")
    ax.plot(grid / 1000, smooth(np.array(cols)), color=color, lw=2.2,
            ls="-" if done == len(runs) else "--",
            label=f"{label}  {done}/{len(runs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274476")
    a = ap.parse_args()

    curves, states = load(a.array)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6))
    fig.subplots_adjust(left=.045, right=.995, top=.74, bottom=.13, wspace=.16)

    for ax, var in zip(axes, VARIANTS):
        draw(ax, curves.get("uniform", []), C_UNI, "uniform", states.get("uniform", []))
        draw(ax, curves.get("per", []), C_PER, "PER", states.get("per", []))
        draw(ax, curves.get(var, []), C_OFF, "bootstrap OFF", states.get(var, []))
        draw(ax, curves.get(f"{var}_bs", []), C_ON, "bootstrap ON",
             states.get(f"{var}_bs", []))
        ax.axvspan(0, DECAY / 1000, color=GRID, alpha=.45, zorder=0)
        ax.text(DECAY / 1000 / 2, 78, "exploring", ha="center", va="top",
                fontsize=7.5, color=INK2, style="italic")
        ax.set_xlim(0, TARGET / 1000); ax.set_ylim(0, 80)
        ax.set_title(LAB[var], fontsize=11, fontweight="bold", color=INK,
                     loc="left", pad=6)
        ax.legend(frameon=False, fontsize=8, loc="upper left",
                  bbox_to_anchor=(0, .95), handlelength=1.6)
        ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
        if ax is axes[0]:
            ax.set_ylabel("evaluation win rate (%)", fontsize=9.5, color=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)

    fig.text(.045, .93, "JaxNav 8×8 fill 0.3 — does bootstrapping the truncated "
             "rollout change learning?", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(.045, .875,
             "Within a panel the ONLY difference between blue and orange is cf_bootstrap "
             "— same seeds, same cell. uniform and PER are the shared controls.",
             fontsize=9.5, color=INK2)
    fig.text(.045, .835,
             "SOLID = every seed finished. DASHED = arm still running, curve stops at the "
             "median seed's progress and is NOT comparable to a finished arm.",
             fontsize=9.5, color=INK2)

    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")

    print(f"\n{'arm':16s} {'seeds':>6s} {'done':>5s} {'median final win%':>18s}")
    for arm in ["uniform", "per"] + [v + s for v in VARIANTS for s in ("", "_bs")]:
        runs, st = curves.get(arm, []), states.get(arm, [])
        if not runs:
            continue
        fin = [np.mean(w[-20:]) for e, w in runs if e[-1] >= TARGET * .96]
        med = f"{np.median(fin):.1f}" if fin else "-- (none finished)"
        print(f"{arm:16s} {len(runs):>6d} {sum(1 for s in st if s=='COMPLETED'):>5d} "
              f"{med:>18s}")


if __name__ == "__main__":
    main()
