"""What the replay buffer is actually DOING -- the instrumentation-run figures.

Reads array 274678 (`manifest_instrument.json`) and draws only quantities that
were IMPOSSIBLE to plot before the instruments landed on 2026-09-03.

SAFE TO RUN MID-SWEEP, and it defends itself two ways:
  * every arm is truncated to the COMMON EPISODE FLOOR across all runs, because
    arms progress at wildly different rates (uniform reaches 250k in ~1.1 h, a CCE
    arm takes ~8.6 h). Comparing "as of now" would rank arms by how fast they run.
  * the episode reached by each arm is printed in the caption, so a partial sweep
    can never be mistaken for a finished one.

Palette: Okabe-Ito, matching the repo's existing figures. The 4-arm categorical set
was checked with the dataviz validator -- worst adjacent CVD separation dE 11.0
(deutan), normal-vision floor 24.2, both passing.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_instrument_figures [--array 274678]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import RUNS, ROOT, resolve

MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_instrument.json")
OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_instrumentation.png")

CCE_ARMS = ["cce_wmean", "cce_max", "cce_only", "cce_wmean_bs"]
LAB = {"cce_wmean": "CCE mul·wmean", "cce_max": "CCE mul·max",
       "cce_only": "CCE only μ=1", "cce_wmean_bs": "CCE wmean +bootstrap",
       "uniform": "uniform", "per": "PER"}
COL = {"cce_wmean": "#0072b2", "cce_max": "#e69f00",
       "cce_only": "#009e73", "cce_wmean_bs": "#d55e00",
       "uniform": "#7a7a78", "per": "#9c8250"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"
SURFACE = "#fcfcfb"
# derived coverage: n_score_sample / (score_interval * n_steps_per_update)
DERIVED_COVERAGE = 64 / (250 * 16)


def load(array_id):
    """arm -> {'ess': [rows...], 'eval': [rows...]} per seed."""
    man = json.load(open(MANIFEST))
    res = resolve(array_id)
    out = {}
    for rec in man.values():
        jid = rec["job_id"]
        if jid not in res:
            continue
        run_dir, _ = res[jid]
        d = os.path.join(RUNS, run_dir)
        rows = {}
        for name in ("ess", "eval"):
            p = os.path.join(d, f"{name}.jsonl")
            if os.path.exists(p):
                rows[name] = [json.loads(l) for l in open(p) if l.strip()]
        if rows:
            out.setdefault(rec["arm"], []).append(rows)
    return out


def _series(seeds, key, floor, source="ess"):
    """Median across seeds of key vs episode, truncated at the common floor."""
    xs, ys = [], []
    for s in seeds:
        rows = [r for r in s.get(source, []) if r.get("episode", 0) <= floor]
        rows = [r for r in rows if r.get(key) is not None]
        if len(rows) < 2:
            continue
        xs.append(np.array([r["episode"] for r in rows], float))
        ys.append(np.array([float(r[key]) for r in rows], float))
    if not xs:
        return None, None
    grid = np.linspace(min(x[0] for x in xs), floor, 300)
    stack = np.vstack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    return grid, np.median(stack, axis=0)


def smooth(y, w=9):
    if y is None or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, w // 2, mode="edge"), k, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274678")
    args = ap.parse_args()

    data = load(args.array)
    if not data:
        print("nothing resolved yet")
        return

    floors = {}
    for arm, seeds in data.items():
        e = [max((r["episode"] for r in s.get("eval", [])), default=0) for s in seeds]
        floors[arm] = min(e) if e else 0
    floor = min(v for v in floors.values() if v > 0)
    print(f"per-arm episodes reached: { {k: v for k, v in sorted(floors.items())} }")
    print(f"common episode floor: {floor:,}")

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.4), facecolor=SURFACE)
    fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.085,
                        wspace=.245, hspace=.34)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=.55, color=GRID, lw=.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    def xfmt(ax):
        ax.set_xlabel("episodes (thousands)", fontsize=9.5, color=INK2)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))

    # ---- 1. COVERAGE -- the headline ------------------------------------
    ax = axes[0, 0]
    finals = []
    for arm in CCE_ARMS:
        if arm not in data:
            continue
        x, y = _series(data[arm], "scored_frac", floor)
        if x is None:
            continue
        ax.plot(x, 100 * smooth(y), lw=2, color=COL[arm], label=LAB[arm])
        finals.append(100 * smooth(y)[-1])
    if finals:
        m = float(np.mean(finals))
        ax.annotate(f"all four arms: {m:.2f}%\nMEASURED, once the buffer is full",
                    xy=(floor, m), xytext=(floor * .42, m + .42),
                    fontsize=9.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.axhline(100 * DERIVED_COVERAGE, color=INK2, ls="--", lw=1.3)
    ax.text(.02, 100 * DERIVED_COVERAGE + .05, "1.60%  DERIVED (a full lifetime)",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK2)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("% of buffer holding a MEASURED score", fontsize=9.5, color=INK2)
    ax.set_title("1.  Replay coverage — measured, not derived", fontsize=11.5,
                 fontweight="bold", color=INK, loc="left", pad=8)
    xfmt(ax)

    # ---- 2. STALENESS ----------------------------------------------------
    ax = axes[0, 1]
    for arm in CCE_ARMS:
        if arm not in data:
            continue
        x, y = _series(data[arm], "staleness_p50", floor)
        if x is None:
            continue
        ax.plot(x, smooth(y), lw=2, color=COL[arm], label=LAB[arm])
        x2, y2 = _series(data[arm], "staleness_p90", floor)
        if x2 is not None:
            ax.plot(x2, smooth(y2), lw=1.1, ls=":", color=COL[arm], alpha=.8)
    ax.set_ylabel("age of the score, in Q-updates", fontsize=9.5, color=INK2)
    ax.set_title("2.  How STALE the scores are  (solid p50, dotted p90)",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    xfmt(ax)

    # ---- 3. ROLLOUTS vs REALITY ------------------------------------------
    ax = axes[0, 2]
    arm = "cce_wmean" if "cce_wmean" in data else CCE_ARMS[0]
    pairs = [("rollout_goal_frac", "win_rate", "#009e73", "GOAL"),
             ("rollout_crash_frac", "crash_rate", "#d55e00", "CRASH")]
    for rk, ek, c, lab in pairs:
        x, y = _series(data[arm], rk, floor)
        if x is not None:
            ax.plot(x, 100 * smooth(y), lw=2.2, color=c, label=f"{lab} — imagined")
        x, y = _series(data[arm], ek, floor, source="eval")
        if x is not None:
            ax.plot(x, 100 * smooth(y), lw=2.2, ls="--", color=c,
                    label=f"{lab} — real")
    ax.set_ylabel("% of episodes / rollouts", fontsize=9.5, color=INK2)
    ax.set_title("3.  What CCE IMAGINES vs what HAPPENS", fontsize=11.5,
                 fontweight="bold", color=INK, loc="left", pad=8)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center")
    xfmt(ax)

    # ---- 4. CCE vs TD spread --------------------------------------------
    ax = axes[1, 0]
    for arm in CCE_ARMS:
        if arm not in data:
            continue
        seeds = data[arm]
        x, sm = _series(seeds, "score_mean", floor)
        _, ss = _series(seeds, "score_std", floor)
        _, tm = _series(seeds, "td_mean", floor)
        _, ts = _series(seeds, "td_std", floor)
        if x is None or sm is None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (ts / tm) / (ss / sm)
        ax.plot(x, smooth(np.nan_to_num(ratio)), lw=2, color=COL[arm], label=LAB[arm])
    ax.axhline(1.0, color=INK2, ls="--", lw=1.2)
    ax.text(.02, 1.15, "equal spread", transform=ax.get_yaxis_transform(),
            fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylabel("cv(TD) / cv(CCE)   — how much louder TD is", fontsize=9.5, color=INK2)
    ax.set_title("4.  TD drowns CCE  (same exponent, unequal spread)",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    xfmt(ax)

    # ---- 5. how EVAL episodes end, every arm -----------------------------
    ax = axes[1, 1]
    order = [a for a in ["uniform", "per"] + CCE_ARMS if a in data]
    w = 0.62
    for i, arm in enumerate(order):
        vals = []
        for key in ("win_rate", "crash_rate", "timeout_rate"):
            _, y = _series(data[arm], key, floor, source="eval")
            vals.append(100 * float(np.mean(y[-20:])) if y is not None else 0.0)
        bot = 0
        for v, c, lab in zip(vals, ["#009e73", "#d55e00", "#b8b6b1"],
                             ["goal", "crash", "timeout"]):
            ax.bar(i, v, w, bottom=bot, color=c,
                   label=lab if i == 0 else None, edgecolor=SURFACE, lw=2)
            if v > 7:
                ax.text(i, bot + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
            bot += v
    ax.set_xticks(range(len(order)))
    _SHORT = {"uniform": "uniform", "per": "PER", "cce_wmean": "wmean",
              "cce_max": "max", "cce_only": "only\nμ=1", "cce_wmean_bs": "wmean\n+bs"}
    ax.set_xticklabels([_SHORT[a] for a in order], fontsize=8.5)
    ax.set_ylabel("% of eval episodes", fontsize=9.5, color=INK2)
    ax.set_title(f"5.  How REAL episodes end  (at {floor/1000:.0f}k episodes)",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center")
    ax.set_ylim(0, 118)

    # ---- 6. length conditional on outcome --------------------------------
    ax = axes[1, 2]
    for arm in ["uniform", "cce_wmean"]:
        if arm not in data:
            continue
        x, y = _series(data[arm], "avg_length_goal", floor, source="eval")
        if x is not None:
            ax.plot(x, smooth(y), lw=2.2, color=COL[arm], label=f"{LAB[arm]} — WINS")
        x, y = _series(data[arm], "avg_length", floor, source="eval")
        if x is not None:
            ax.plot(x, smooth(y), lw=1.4, ls=":", color=COL[arm],
                    label=f"{LAB[arm]} — all episodes")
    ax.set_ylabel("episode length (steps)", fontsize=9.5, color=INK2)
    ax.set_title("6.  Length of the WINS  (pooled average hides it)",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax.legend(fontsize=8, frameon=False, loc="best")
    xfmt(ax)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(fontsize=8, frameon=False, loc="best")

    fig.text(.055, .955, "What the replay buffer is actually doing",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(.055, .906,
             f"JaxNav 8×8 fill 0.3 · array {args.array} · 3 seeds/arm · median across seeds · "
             f"every arm truncated to the common floor of {floor:,} episodes of 250,000.",
             fontsize=10.5, color=INK2)
    fig.text(.055, .868,
             "MECHANISM run, not an outcome run: n=3 cannot show that any arm LEARNS better "
             "(paired seed sd is 30pp; a 5pp effect needs ~290 pairs).",
             fontsize=10.5, color=INK2)
    fig.text(.055, .830,
             "Panels 1, 2, 3 and 6 plot quantities that did not exist before the "
             "instruments landed on 2026-09-03.",
             fontsize=9.5, color=INK2, style="italic")

    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
