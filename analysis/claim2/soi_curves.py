"""Training curves for the score-on-insert x collision-reward sweep (arrays 275761/275791).

Top row: win rate, one panel per REWARD CELL. Arms are compared only within a cell --
coll_rew=-1 is a different learning problem, so it gets its own baselines from the same
sweep (n=5), while the coll_rew=0 cell uses the pooled default-environment runs
(uniform n=26, PER n=18; config verified identical).

Bottom row: fraction of buffer scores that are EXACTLY zero, per CCE arm, over training.
That is the structural half of the result and it is already readable, where the win
rates are not.

Colours mean the same thing in every panel: grey = baselines (solid uniform, dotted PER),
teal = CCE H=60, purple = CCE H=200.

CCE arms are STILL RUNNING and their lines stop where the runs are. A CCE line ending
below a finished baseline means "not finished", not "losing". Left of the epsilon marker
is exploration, not an algorithm comparison.
"""
import json
import os
import subprocess

import numpy as np

import greedy_learning_curves as G

OUT = os.path.join(G.WT, "docs/figures/real/claim2/jaxnav")
MANIFEST = os.path.join(G.DATA, "manifest_score_on_insert.json")
ARRAYS = "275761,275791"

CELLS = [("r0",   "crash reward = 0   (the environment as it has always been)"),
         ("rneg", "crash reward = -1   (crash now worse than timing out)")]
BASE_STYLE = {"uniform": ("#5f6b73", "-", 2.0), "per": ("#5f6b73", ":", 2.0)}
SOI_STYLE = {60: "#0d7c6b", 200: "#7b4fa8"}


def _zero_series():
    """Per CCE arm: list of (episodes, zero_fraction) arrays, one per run."""
    man = json.load(open(MANIFEST))
    out = subprocess.run(["sacct", "-j", ARRAYS, "-X", "-n", "-P", "--format=JobID,JobIDRaw"],
                         capture_output=True, text=True).stdout
    raw = {l.split("|")[0]: l.split("|")[1] for l in out.strip().splitlines() if "|" in l}
    series = {}
    for v in man.values():
        if not v["arm"].startswith("soi_"):
            continue
        f = os.path.join(G.RUNS, raw.get(v["job_id"], v["job_id"]), "ess.jsonl")
        if not os.path.exists(f):
            continue
        pts = []
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("scored_n") and r.get("episode") is not None:
                pts.append((r["episode"], 100 * r["score_n_zero"] / r["scored_n"]))
        if len(pts) > 3:
            series.setdefault(v["arm"], []).append(np.array(pts))
    return series


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c = G.collect(G._label_soi)
    zs = _zero_series()

    fig = plt.figure(figsize=(15.5, 10.2), dpi=150)
    top = [fig.add_axes([.055, .455, .385, .40]), fig.add_axes([.535, .455, .385, .40])]
    bot = [fig.add_axes([.055, .075, .385, .28]), fig.add_axes([.535, .075, .385, .28])]

    for ax, zax, (tag, title) in zip(top, bot, CELLS):
        # baselines first, with an IQR band on uniform so the seed spread is visible
        for algo in ("uniform", "per"):
            k = f"{algo}_{tag}"
            if k not in c:
                continue
            col, ls, lw = BASE_STYLE[algo]
            if algo == "uniform":
                ax.fill_between(G.GRID, c[k]["lo"], c[k]["hi"], color=col, alpha=.12, lw=0)
            ax.plot(G.GRID, c[k]["med"], color=col, ls=ls, lw=lw)
            fin = c[k]["med"][~np.isnan(c[k]["med"])]
            if fin.size:
                ax.text(253_000, fin[-1], f"{'PER' if algo=='per' else algo}  n={c[k]['n']}",
                        fontsize=8.5, color=col, va="center")
        for h, colr in SOI_STYLE.items():
            k = f"soi_h{h}_{tag}"
            if k not in c:
                continue
            ax.fill_between(G.GRID, c[k]["lo"], c[k]["hi"], color=colr, alpha=.14, lw=0)
            ax.plot(G.GRID, c[k]["med"], color=colr, lw=2.6)
            m = c[k]["med"]; ok = np.where(~np.isnan(m))[0]
            if ok.size:
                ax.plot(G.GRID[ok[-1]], m[ok[-1]], "o", color=colr, ms=6, zorder=5)
                ax.text(G.GRID[ok[-1]] + 4_000, m[ok[-1]] + (6.5 if h == 200 else -4.0),
                        f"H={h}  (running, {G.GRID[ok[-1]]//1000}k)", fontsize=8.5, color=colr)
        ax.axvline(G.EPS_DECAY, color="#a8341c", lw=1.1, ls="--")
        ax.set_xlim(0, 250_000); ax.set_ylim(0, 80)
        ax.set_title(title, fontsize=11.5, loc="left", pad=8)
        ax.set_ylabel("win rate  (%)", fontsize=10)
        ax.grid(alpha=.18, lw=.7)
        ax.set_xticks([0, 50_000, 100_000, 150_000, 200_000, 250_000])
        ax.set_xticklabels(["0", "50k", "100k", "150k", "200k", "250k"])

        # dead-score fraction, every run faint, arm median bold
        for h, colr in SOI_STYLE.items():
            arm = f"soi_h{h}_{tag}"
            runs = zs.get(arm, [])
            for r in runs:
                zax.plot(r[:, 0], r[:, 1], color=colr, lw=.8, alpha=.28)
            if runs:
                grid = np.arange(0, max(r[-1, 0] for r in runs) + 1, 1_000)
                M = np.array([np.interp(grid, r[:, 0], r[:, 1], left=np.nan, right=np.nan)
                              for r in runs])
                with np.errstate(all="ignore"):
                    med = np.nanmedian(M, axis=0)
                med[np.sum(~np.isnan(M), axis=0) < 3] = np.nan
                zax.plot(grid, med, color=colr, lw=2.4)
                okz = np.where(~np.isnan(med))[0]
                if okz.size:
                    zax.text(grid[okz[-1]] + 1_500, med[okz[-1]], f"H={h}  {med[okz[-1]]:.0f}%",
                             fontsize=9, color=colr, va="center")
        zax.set_xlim(0, 60_000); zax.set_ylim(0, 100)
        zax.set_ylabel("scores EXACTLY 0  (%)", fontsize=10)
        zax.set_xlabel("episode", fontsize=10)
        zax.grid(alpha=.18, lw=.7)
        zax.set_xticks([0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
        zax.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k", "60k"])

    top[0].text(G.EPS_DECAY + 3_000, 74, "epsilon decay ends\nleft = exploration",
                fontsize=8, color="#a8341c", va="top")
    top[1].set_yticklabels([]); top[1].set_ylabel("")
    bot[1].set_yticklabels([]); bot[1].set_ylabel("")
    bot[0].set_title("dead scores over training  (every run faint, median bold)",
                     fontsize=10.5, loc="left", pad=6)
    bot[1].set_title("same, crash reward = -1", fontsize=10.5, loc="left", pad=6)

    handles = [Line2D([], [], color="#5f6b73", lw=2, label="uniform"),
               Line2D([], [], color="#5f6b73", lw=2, ls=":", label="PER"),
               Line2D([], [], color=SOI_STYLE[60], lw=2.6, label="CCE score-on-insert, H=60"),
               Line2D([], [], color=SOI_STYLE[200], lw=2.6, label="CCE score-on-insert, H=200")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.92, .905), ncol=4,
               fontsize=9.5, frameon=False)
    fig.text(.055, .955, "Score-on-insert (100% coverage) x crash reward   —   JaxNav 8x8, fill 0.3",
             fontsize=13.5, weight="bold")
    fig.text(.055, .928, "CCE arms STILL RUNNING: lines stop where the runs are. "
             "Compare arms only inside one panel — crash = -1 is a different problem.",
             fontsize=9.5, color="#33444d")

    p = os.path.join(OUT, "soi_curves.png")
    fig.savefig(p, facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
