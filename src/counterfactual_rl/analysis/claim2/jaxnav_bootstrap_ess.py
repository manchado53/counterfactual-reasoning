"""Replay-signal health across training for the bootstrap sweep.

The win-rate figure says whether bootstrap changed the OUTCOME. This one says
whether it changed the THING WE ACTUALLY EDITED -- the score the buffer samples
by -- and it reads the live buffer the trainer sampled from, not a rescored
checkpoint.

Three rows, from `ess.jsonl` (written once per eval by
`consequence_dqn_vectorized._log_priority_diagnostics`):

    score_cv          std/mean of the score across the buffer -- can it rank
                      anything at all? A raw zero-count is useless here: a new
                      transition inherits the running MEAN (consequence_buffers
                      .py:102), so literal zeros vanish whether or not the score
                      is informative.
    ess_frac          effective sample size / n. 1.0 = sampling uniformly,
                      i.e. the score is not differentiating the buffer at all
                      -- whether because it is flat at ZERO or flat at HIGH.
                      Magnitude and spread are different things.
    score_mean        where the score sits. Rises with bootstrap by
                      construction; shown so a change in ess_frac can be read
                      against it rather than confused with it.

uniform and PER never compute a consequence score, so they have no ess.jsonl and
do not appear -- the comparison here is strictly bootstrap OFF vs ON.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_bootstrap_ess [--array 274476]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import resolve, MANIFEST, RUNS, ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_bootstrap_ess.png")

TARGET, DECAY = 250_000, 62_500
VARIANTS = ["cce_wmean", "cce_max", "cce_add", "cce_only"]
LAB = {"cce_wmean": "CCE mul·wmean", "cce_max": "CCE mul·max",
       "cce_add": "CCE add μ=.25", "cce_only": "CCE only μ=1"}
# score_zero_frac is NOT plotted, deliberately. A new transition is initialised to
# the running MEAN of existing scores (consequence_buffers.py:102), not to zero, so
# after the first nonzero score almost nothing in the buffer holds a literal 0 and
# the field reads ~0 regardless of whether the score is informative. The honest
# measure of "can this score rank anything" is its RELATIVE spread, std/mean.
ROWS = [("score_cv",   "score spread  (std / mean)", (None, None), True),
        ("ess_frac",   "ess_frac (1.0 = uniform)",   (0.5, 1.02), False),
        ("score_mean", "mean CCE score",             (None, None), False)]
C_OFF, C_ON = "#0072b2", "#d55e00"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"


def load(array_id):
    """(arm) -> [(episodes, {field: series}), ...] one entry per seed."""
    man = json.load(open(MANIFEST))
    resolved = resolve(array_id)
    out = {}
    for key, rec in man.items():
        jid = rec["job_id"]
        if jid not in resolved:
            continue
        run_dir, _ = resolved[jid]
        f = os.path.join(RUNS, run_dir, "ess.jsonl")
        if not os.path.exists(f):
            continue                      # uniform / PER have no consequence buffer
        eps, cols = [], {k: [] for k, _, _, _ in ROWS}
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                  # a torn last line while the run is live
            eps.append(r.get("episode", 0))
            m, sd = r.get("score_mean", np.nan), r.get("score_std", np.nan)
            r["score_cv"] = sd / m if m else np.nan
            for k, _, _, _ in ROWS:
                cols[k].append(r.get(k, np.nan))
        if len(eps) > 3:
            out.setdefault(rec["arm"], []).append(
                (np.array(eps, float), {k: np.array(v, float) for k, v in cols.items()}))
    return out


def smooth(y, w=9):
    if len(y) < w:
        return y
    pad = w // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(w) / w, mode="valid")[:len(y)]


def band(ax, runs, field, color, label):
    """Median across seeds, with the seed spread shaded."""
    if not runs:
        return
    ends = sorted(e[-1] for e, _ in runs)
    hi = ends[len(ends) // 2]
    if hi < 2000:
        return
    grid = np.linspace(250, hi, 200)
    stack = np.array([np.interp(grid, e, c[field]) for e, c in runs if e[-1] >= grid[0]])
    med = smooth(np.median(stack, axis=0))
    lo = smooth(np.percentile(stack, 25, axis=0))
    up = smooth(np.percentile(stack, 75, axis=0))
    ax.fill_between(grid / 1000, lo, up, color=color, alpha=.16, lw=0)
    ax.plot(grid / 1000, med, color=color, lw=2.1, label=f"{label}  n={len(runs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274476")
    a = ap.parse_args()

    data = load(a.array)
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(17.5, 10.6), sharex=True)
    # top leaves room for a 6-line header; the panel titles sit just under it
    fig.subplots_adjust(left=.055, right=.99, top=.815, bottom=.055,
                        wspace=.16, hspace=.16)

    for col, var in enumerate(VARIANTS):
        for row, (field, ylab, ylim, pct) in enumerate(ROWS):
            ax = axes[row][col]
            band(ax, data.get(var, []), field, C_OFF, "bootstrap OFF")
            band(ax, data.get(f"{var}_bs", []), field, C_ON, "bootstrap ON")
            ax.axvspan(0, DECAY / 1000, color=GRID, alpha=.45, zorder=0)
            ax.set_xlim(0, TARGET / 1000)
            if pct:
                ax.set_yscale("log")
            if ylim[0] is not None:
                ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(LAB[var], fontsize=11, fontweight="bold",
                             color=INK, loc="left", pad=6)
                ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.6)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
            if row == len(ROWS) - 1:
                ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8.5)
            ax.grid(axis="y", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)

    fig.text(.055, .968, "JaxNav 8\u00d78 fill 0.3 \u2014 what bootstrapping did to the "
             "replay signal itself", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(.055, .936,
             "Read from the LIVE buffer every eval (ess.jsonl), not from rescored "
             "checkpoints. Line = median across seeds, band = interquartile spread.",
             fontsize=9.5, color=INK2)
    fig.text(.055, .913,
             "Row 1 (log): can the score rank anything \u2014 std/mean across the buffer.  "
             "Row 2: does it actually concentrate replay?",
             fontsize=9.5, color=INK2)
    fig.text(.055, .890,
             "ess_frac 1.0 = uniform sampling, which a uniformly-HIGH score gives just as "
             "surely as a uniformly-zero one.",
             fontsize=9.5, color=INK2)
    fig.text(.055, .862,
             "CAVEAT ON ALL THREE ROWS: only ~1.6% of the buffer ever receives a measured "
             "score (131 scored per 8,192 added, 100k capacity).",
             fontsize=9.5, color="#8a3324")
    fig.text(.055, .839,
             "The other ~98.4% carries the running MEAN it inherited on insertion "
             "(consequence_buffers.py:102) \u2014 a constant, which cannot rank anything.",
             fontsize=9.5, color="#8a3324")
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")

    print(f"\n{'arm':16s} {'n':>3s}  " + "  ".join(f"{k:>22s}" for k, _, _, _ in ROWS))
    print(f"{'':16s} {'':>3s}  " + "  ".join(f"{'(first -> last)':>22s}" for _ in ROWS))
    for var in VARIANTS:
        for arm in (var, f"{var}_bs"):
            runs = data.get(arm, [])
            if not runs:
                continue
            cells = []
            for field, _, _, _ in ROWS:
                first = np.median([c[field][:5].mean() for _, c in runs])
                last = np.median([c[field][-20:].mean() for _, c in runs])
                cells.append(f"{first:9.4f} -> {last:8.4f}")
            print(f"{arm:16s} {len(runs):>3d}  " + "  ".join(cells))
        print()


if __name__ == "__main__":
    main()
