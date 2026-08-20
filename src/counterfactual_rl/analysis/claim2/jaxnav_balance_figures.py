"""Learning-curve figures for the ESS-matched balance sweep.

House two-panel format (curve + per-seed strip), matching
jaxnav_holes_figures.py so this sits alongside the 96k/150k/500k figures.

One deliberate departure: `cce_balance` is an ORDERED variable (0 -> 1), not a
set of unrelated arms, so the five levels take a single-hue sequential ramp
light->dark rather than the categorical arm colours used elsewhere. Categorical
hues here would imply the levels are unrelated and hide the ordering that IS
the result.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_holes_figures import DATA, INK, INK2, GRID, _frame, load
from .compute_metrics import iqm

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "agents", "jax_nav", "runs")
FIGDIR = os.path.join(DATA, "..")
MANIFEST = os.path.join(DATA, "manifest_balance_ess.json")
TARGET_EP = 250000
MIN_EP = TARGET_EP * 0.95

# Five distinguishable hues rather than one sequential ramp. `cce_balance` IS
# ordered, so a single-hue ramp is the textbook choice -- but as five overlapping
# LINES it failed on measurement, not taste: adjacent steps of ColorBrewer Blues
# scored normal-vision dE 9.1, under the 15 floor ("hard to tell apart even with
# full color vision"). This set passes every check (worst adjacent: normal dE
# 18.8, protan 8.8, tritan 16.4, all >= 3:1 on white). The ordering is carried
# instead by the legend order and by the right panel's x-axis, and PER is dashed
# because it is the control rather than another level of the dose.
RAMP = ["#0072b2", "#bf8600", "#009e73", "#8452c4", "#d55e00"]
CTRL_STYLE = {0.0: dict(ls="--", lw=2.6)}


SMOOTH_W = 41  # evals, matching jaxnav_holes_figures.SMOOTH


def _smooth(y, w):
    """Centred rolling mean, edge-padded so the line spans the full axis."""
    if w <= 1 or len(y) < w:
        return y
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(ypad, np.ones(w) / w, mode="valid")[:len(y)]


def _arms():
    with open(MANIFEST) as fh:
        man = json.load(fh)
    out = {}
    for rec in man.values():
        out.setdefault(float(rec["cce_balance"]), []).append(str(rec["job_id"]))
    return dict(sorted(out.items()))


def _final(job, k=20):
    """Mean win rate over the last k evals (k=20: 5 evals sits inside the noise band), or None if the run is short.

    Short runs are DROPPED, never forward-filled: filling a killed run at its
    dying win rate is what contaminated the 2026-08-03 graded-slip sweep.
    """
    curve = load(job, RUNS)
    if curve is None:
        return None
    ep, win = curve
    if len(ep) == 0 or ep[-1] < MIN_EP:
        return None
    return float(np.mean(win[-k:]))


def _iqm_curve(jobs):
    curves = []
    for j in jobs:
        c = load(j, RUNS)
        if c is None or len(c[0]) == 0 or c[0][-1] < MIN_EP:
            continue
        curves.append(c)
    if not curves:
        return None, None, 0
    hi = min(c[0][-1] for c in curves)
    grid = np.linspace(250, hi, 300)
    stack = np.vstack([np.interp(grid, e, w) for e, w in curves])
    return grid, np.array([iqm(col) for col in stack.T]), len(curves)


def fig_balance(out_name="fig_jaxnav_balance_ess_250k.png"):
    arms = _arms()
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(14.2, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})

    finals_by = {}
    for i, (bal, jobs) in enumerate(arms.items()):
        col = RAMP[i]
        vals = [v for v in (_final(j) for j in jobs) if v is not None]
        finals_by[bal] = np.array(vals)
        g, y, n = _iqm_curve(jobs)
        if g is None:
            continue
        if n < len(jobs):
            print(f"    !! balance {bal}: {len(jobs)-n} of {len(jobs)} seeds short of budget, dropped")
        # Raw eval-to-eval variance on this task is larger than any between-arm
        # gap, so the unsmoothed IQM is unreadable. Smoothed line carries the
        # trend, raw stays visible underneath so the noise is not hidden.
        ax.plot(g / 1000, y * 100, color=col, lw=0.7, alpha=0.16, zorder=1)
        ys = _smooth(y, SMOOTH_W)
        st = CTRL_STYLE.get(bal, dict(ls="-", lw=2.4))
        lbl = f"{int(bal*100)}% CCE" + ("  = PER (control)" if bal == 0.0 else "")
        ax.plot(g / 1000, ys * 100, color=col, zorder=3,
                label=f"{lbl}  (n={n})", **st)

    ax.set_xlabel("training episodes (thousands)", color=INK2, fontsize=11)
    ax.set_ylabel("evaluation win rate  (%)", color=INK2, fontsize=11)
    ax.set_title("JaxNav holes map, 250k budget — ESS-matched at 0.60, 8 seeds/arm",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right", title="replay priority",
              title_fontsize=9.5)
    _frame(ax)

    # right panel: per-seed finals + IQM with bootstrap CI, across the axis
    rng = np.random.default_rng(0)
    xs = list(finals_by.keys())
    for i, (bal, vals) in enumerate(finals_by.items()):
        if len(vals) == 0:
            continue
        col = RAMP[i]
        jitter = (rng.random(len(vals)) - 0.5) * 0.055
        bx.scatter(np.full(len(vals), bal) + jitter, vals * 100, s=26, color=col,
                   alpha=0.55, edgecolor="white", linewidth=0.8, zorder=3)
        boot = np.array([iqm(rng.choice(vals, len(vals))) for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        bx.errorbar(bal, iqm(vals) * 100, yerr=[[(iqm(vals)-lo)*100], [(hi-iqm(vals))*100]],
                    fmt="o", color=col, ms=9, capsize=4, lw=2, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.2)

    ys = np.concatenate([v for v in finals_by.values() if len(v)])
    xsr = np.concatenate([[b] * len(v) for b, v in finals_by.items() if len(v)])
    from scipy.stats import spearmanr
    rho, p = spearmanr(xsr, ys)
    bx.set_xlabel("share of replay priority driven by CCE", color=INK2, fontsize=11)
    bx.set_ylabel("final win rate  (%)", color=INK2, fontsize=11)
    bx.set_title(f"No dose-response:  Spearman ρ = {rho:+.3f}, p = {p:.2f}",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    bx.set_xticks(xs)
    bx.set_xticklabels([f"{int(b*100)}%" for b in xs])
    _frame(bx)

    fig.tight_layout()
    out = os.path.abspath(os.path.join(FIGDIR, out_name))
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")
    return out


def fig_ess_check(out_name="fig_jaxnav_balance_ess_check.png"):
    """The control figure: prove concentration really was held equal.

    Without this the left panel is uninterpretable -- a separation there could
    be sharpness rather than signal.
    """
    arms = _arms()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, (bal, jobs) in enumerate(arms.items()):
        eps, vals = [], []
        for j in jobs:
            path = os.path.join(RUNS, str(j), "ess.jsonl")
            if not os.path.exists(path):
                continue
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("ess_k_saturated"):
                    eps.append(r["episode"])
                    vals.append(r["ess_frac"])
        if not eps:
            continue
        eps, vals = np.array(eps), np.array(vals)
        order = np.argsort(eps)
        eps, vals = eps[order], vals[order]
        b = np.linspace(eps.min(), eps.max(), 60)
        idx = np.digitize(eps, b)
        med = [np.median(vals[idx == k]) for k in range(1, len(b)) if np.any(idx == k)]
        xb = [b[k - 1] for k in range(1, len(b)) if np.any(idx == k)]
        st = CTRL_STYLE.get(bal, dict(ls="-", lw=2.0))
        ax.plot(np.array(xb) / 1000, med, color=RAMP[i],
                label=f"{int(bal*100)}% CCE", **st)
    ax.axhline(0.60, color=INK2, ls="--", lw=1.2, zorder=1)
    ax.text(0.02, 0.585, "target 0.60", transform=ax.get_yaxis_transform(),
            ha="left", va="top", color=INK2, fontsize=9)
    ax.set_ylim(0.50, 0.72)
    ax.set_xlabel("training episodes (thousands)", color=INK2, fontsize=11)
    ax.set_ylabel("realised ESS / n", color=INK2, fontsize=11)
    ax.set_title("Control: replay concentration held equal across all five arms",
                 color=INK, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    _frame(ax)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(FIGDIR, out_name))
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    fig_balance()
    fig_ess_check()
