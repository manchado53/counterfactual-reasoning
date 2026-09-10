"""Win-rate curves for the greedy-coverage sweep against every fill-0.3 baseline on disk.

Writes a PNG for the repo and an inline-SVG fragment for the explainer page.

TWO THINGS THIS PLOT MUST NOT BE MISREAD AS:
  1. The greedy arms are STILL RUNNING and stop where they stop. Their lines ending
     lower than a completed baseline means "not finished", not "losing".
  2. epsilon decays over the first 62,500 episodes. Anything left of that marker is
     dominated by exploration and is not an algorithm comparison. A vertical rule is
     drawn there for exactly this reason -- an earlier read of this sweep compared
     arms 36,000 episodes apart and looked alarming for no reason.

Curves are per-arm MEDIANS over seeds, on a common episode grid. The uniform IQR band
is drawn because the seed spread is the reason n=5 resolves only ~35 pp.
"""
import collections
import json
import glob
import os
import subprocess

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
RUNS = os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs")
DATA = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data")
OUTDIR = os.path.join(WT, "docs/figures/real/claim2/jaxnav")
EPS_DECAY = 62_500
GRID = np.arange(0, 250_001, 2_500)

STYLE = [
    ("uniform",         "uniform",              "#7c8a94", "-",  1.6),
    ("per",             "PER",                  "#7c8a94", ":",  1.6),
    ("cce_base_bs",     "CCE shipped (0.86%)",  "#3f5763", "-",  2.0),
    ("greedy_base_max", "greedy_base max",      "#9ec6bd", "-",  1.8),
    ("greedy_base_wm",  "greedy_base wmean",    "#9ec6bd", "--", 1.8),
    ("greedy_cov_max",  "greedy_cov max (16%)", "#0d7c6b", "-",  2.4),
    ("greedy_cov_wm",   "greedy_cov wmean",     "#0d7c6b", "--", 2.4),
    # same ~16% coverage as greedy_cov, reached by scoring 20x more often instead of
    # 20x bigger. Distinct colour on purpose: if it lands on greedy_cov, the ROUTE to
    # coverage does not matter; if it does not, WHEN scores are written does.
    ("greedy_freq_max", "greedy_freq max (16%)", "#7b4fa8", "-",  2.2),
]



def _smooth(v, w=5):
    """Rolling median, window w, SHRINKING at the edges.

    Not convolution: a zero-padded 'same' convolution invents a cliff at both ends,
    which cost a figure earlier in this project. Windows here are clipped to the data
    that exists, so the first and last points are medians of fewer samples, never of
    imaginary zeros.
    """
    out = np.full_like(v, np.nan, dtype=float)
    h = w // 2
    for i in range(len(v)):
        seg = v[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def _label(cfg, arm):
    if arm and arm.startswith("greedy"):
        return arm
    algo = cfg.get("algorithm")
    if algo == "dqn":
        return "per"
    if algo == "dqn-uniform":
        return "uniform"
    # only the shipped multiplicative CCE at full capacity, bootstrapped, R=20
    if (cfg.get("buffer_capacity") or 100_000) != 100_000:
        return None
    if cfg.get("score_interval") != 250 or cfg.get("cf_n_rollouts") != 20:
        return None
    if cfg.get("priority_mixing") != "multiplicative" or not cfg.get("cf_bootstrap"):
        return None
    # PIN the aggregation. The bootstrap sweep ran BOTH max and weighted_mean at seeds
    # 0-4, so without this the one-run-per-seed dedup silently blends two different
    # configurations into a single "CCE shipped" line. weighted_mean is what the paper
    # describes, so that is the one this curve claims to be.
    if cfg.get("consequence_aggregation") != "weighted_mean":
        return None
    return "cce_base_bs"


def collect(label_fn=None, seeds=None):
    jobs = {}
    for p in glob.glob(os.path.join(DATA, "manifest_*.json")):
        if "smoke" in p:
            continue
        for k, v in json.load(open(p)).items():
            if not isinstance(v, dict):
                continue
            cfg = v if "algorithm" in v else v.get("config", {})
            # map_size MUST be checked. The factorial swept {8x8, 11x11} x {0.1,0.3,0.5},
            # so filtering on fill alone pulls 11x11 runs into every baseline. 11x11 is a
            # DIFFERENT problem, not a harder one -- max_steps=200 is fixed across map
            # sizes, so there is less time to cross a bigger world. Measured, uniform at
            # fill 0.3 / 250k: 8x8 median 60.2%, 11x11 median 5.4%. Mixing them dragged
            # the uniform, PER and cce_base curves down by tens of points.
            if (cfg.get("fill") != 0.3 or cfg.get("n_episodes") != 250_000
                    or list(cfg.get("map_size") or []) != [8, 8]):
                continue
            jobs[v.get("job_id", k)] = (cfg, v.get("arm"))
    arrays = sorted({j.split("_")[0] for j in jobs})
    out = subprocess.run(["sacct", "-j", ",".join(arrays), "-X", "-n", "-P",
                          "--format=JobID,JobIDRaw,State"],
                         capture_output=True, text=True).stdout
    raw, state = {}, {}
    for line in out.strip().splitlines():
        p = line.split("|")
        if len(p) >= 3:
            raw[p[0]], state[p[0]] = p[1], p[2].split()[0]

    series = collections.defaultdict(list)
    best = {}   # (label, seed) -> (rank, points), used only when `seeds` is given
    for jid, (cfg, arm) in jobs.items():
        lab = (label_fn or _label)(cfg, arm)
        # CANCELLED is admitted for CURVES ONLY, and only because the cancellation
        # was an operator decision (freeing GPUs on 2026-09-10), not anything about
        # the run. FAILED / TIMEOUT / NODE_FAIL stay excluded: keying off "has a
        # metrics.log" once counted crashed runs as "never solved" and biased the
        # arms unevenly. A curve-vs-episode is unbiased by early truncation; a
        # FINAL-value comparison is NOT, because cancelled runs stopped at
        # 71-96% of training. Never take a final number from a CANCELLED run.
        if lab is None or state.get(jid) not in ("COMPLETED", "RUNNING", "CANCELLED"):
            continue
        # Restrict every arm to the SAME seed indices. Without this, uniform's median is
        # over 26 seeds and the new arms' over 5, so the baselines look smoother purely
        # because they average more runs. NOTE this equalises NOISE, not worlds: arms
        # share one RNG stream and CCE consumes extra keys for its rollouts, so seed s is
        # a different environment sequence in each arm (measured: cross-arm correlation
        # over 12 pairs at n=25 scatters around zero).
        if seeds is not None and cfg.get("seed") not in seeds:
            continue
        rid = raw.get(jid, jid)
        f = os.path.join(RUNS, rid, "eval.jsonl")
        pts = None
        if os.path.exists(f):
            rows = [json.loads(x) for x in open(f) if x.strip()]
            if len(rows) > 5:
                pts = np.array([[r["episode"], 100 * r["win_rate"]] for r in rows])
        else:
            f = os.path.join(RUNS, rid, "metrics.log")
            if os.path.exists(f):
                acc = []
                for line in open(f):
                    if line.startswith("#") or not line.strip():
                        continue
                    t = line.split()
                    if len(t) >= 4:
                        try:
                            acc.append([float(t[0]), float(t[3].rstrip("%"))])
                        except ValueError:
                            pass
                if len(acc) > 5:
                    pts = np.array(acc)
        if pts is not None:
            # One run per (arm, seed). The baselines appear in SEVERAL sweeps -- uniform
            # alone is in factorial, bootstrap, instrument and capacity_outcome -- so
            # filtering to seeds 0-4 still matches ~4 runs per seed and the "n=5" would
            # be a lie. Keep the run that reached furthest, preferring COMPLETED.
            if seeds is None:
                series[lab].append(pts)
            else:
                key = (lab, cfg.get("seed"))
                better = (state.get(jid) == "COMPLETED", pts[-1, 0])
                if key not in best or better > best[key][0]:
                    best[key] = (better, pts)

    if seeds is not None:
        series = collections.defaultdict(list)
        for (lab, _seed), (_rank, pts) in best.items():
            series[lab].append(pts)

    curves = {}
    for lab, runs in series.items():
        M = np.array([np.interp(GRID, r[:, 0], r[:, 1], left=np.nan, right=np.nan)
                      for r in runs])
        n_at = np.sum(~np.isnan(M), axis=0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(M, axis=0)
            lo = np.nanpercentile(M, 25, axis=0)
            hi = np.nanpercentile(M, 75, axis=0)
        # a median over fewer than 3 of 5 seeds is an edge artifact, not a curve
        for a in (med, lo, hi):
            a[n_at < 3] = np.nan
        med, lo, hi = _smooth(med), _smooth(lo), _smooth(hi)
        curves[lab] = dict(med=med, lo=lo, hi=hi, n=M.shape[0],
                           reach=float(max(r[-1, 0] for r in runs)))
    return curves


# ── Second arm-set: the CAPACITY collapse (issues #17, #22) ──────────────────────
# Everything here shrinks the buffer to force coverage up, which is the confound the
# greedy arms exist to avoid: below the 8,192-transition insert burst, part of every
# chunk is overwritten before a single gradient step sees it. cce_8192 is the exact
# break-even (largest shrink that discards nothing); cce_8192_i4 adds 4x scoring
# frequency on top of it and is the only 250k outcome test of the frequency lever.
CAP_STYLE = [
    ("uniform",      "uniform  (no CCE)",          "#7c8a94", "-",  1.8),
    ("cce_base_bs",  "CCE, buffer 100,000",        "#3f5763", "-",  2.2),
    ("cce_8192",     "buffer 8,192  (break-even)", "#c98a5a", "-",  1.8),
    ("cce_8192_i4",  "buffer 8,192, 4x scoring",   "#a8341c", "-",  2.2),
    ("cce_2500",     "buffer 2,500",               "#8a6a9c", "-",  1.5),
    ("cce_1000",     "buffer 1,000",               "#8a6a9c", "--", 1.5),
    ("cce_500",      "buffer 500",                 "#8a6a9c", ":",  1.5),
]


def _label_capacity(cfg, arm):
    if cfg.get("algorithm") == "dqn-uniform":
        return "uniform"
    if cfg.get("algorithm") != "consequence-dqn":
        return None
    if arm in ("cce_8192", "cce_8192_i4", "cce_2500", "cce_1000", "cce_500"):
        return arm
    # the full-capacity reference: shipped multiplicative CCE, bootstrapped, R=20
    if (cfg.get("buffer_capacity") or 100_000) != 100_000:
        return None
    if cfg.get("score_interval") != 250 or cfg.get("cf_n_rollouts") != 20:
        return None
    if cfg.get("priority_mixing") != "multiplicative" or not cfg.get("cf_bootstrap"):
        return None
    return "cce_base_bs"


# ── Third arm-set: PRIORITY NORMALISATION (issue #16) ────────────────────────────
# Jeremy 08/20: "equal exponents do not imply equal contribution". All three sit on the
# greedy_cov config (16% coverage) and differ ONLY in normalise_priorities, so 'none' is
# greedy_cov_max itself and needs no separate run.
NORM_STYLE = [
    ("uniform",        "uniform  (no CCE)",     "#5a5a5a", "-",  1.8),
    ("greedy_cov_max", "none  (shipped)",       "#1f6fb2", "-",  2.5),
    ("cov_minmax_max", "min-max  (Jeremy's ask)", "#e07b39", "-", 2.5),
    ("cov_rank_max",   "rank  (issue #16)",     "#c0392b", "-",  2.5),
]


def _label_norm(cfg, arm):
    if cfg.get("algorithm") == "dqn-uniform":
        return "uniform"
    if arm in ("greedy_cov_max", "cov_minmax_max", "cov_rank_max"):
        return arm
    return None


def write_png(curves, style=None, title=None, out_name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    u = curves.get("uniform")
    if u is not None:
        ax.fill_between(GRID, u["lo"], u["hi"], color="#7c8a94", alpha=.14, lw=0,
                        label="uniform IQR (seed spread)")
    for key, lab, colour, ls, lw in (style or STYLE):
        c = curves.get(key)
        if c is None:
            continue
        ax.plot(GRID, c["med"], color=colour, ls=ls, lw=lw,
                label=f"{lab}  n={c['n']}")
    ax.axvline(EPS_DECAY, color="#a8341c", lw=1.1, ls="--")
    ax.text(EPS_DECAY + 3000, 2, "epsilon decay ends\nleft of here = exploration",
            fontsize=8, color="#a8341c", va="bottom")
    ax.set_xlabel("episode")
    ax.set_ylabel("win rate  (%)")
    ax.set_title(title or ("JaxNav 8x8 fill 0.3 — win rate over training  (rolling median, 5 evals)\n"
                 "greedy arms STILL RUNNING; lines stop where the runs are now"),
                 fontsize=11, loc="left")
    ax.set_xlim(0, 250_000)
    ax.set_ylim(0, 80)
    ax.grid(alpha=.18, lw=.7)
    ax.legend(fontsize=8, loc="upper left", framealpha=.9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, out_name or "greedy_learning_curves.png")
    fig.savefig(out)
    print("wrote", out)


def write_svg(curves):
    """Inline-SVG fragment for the explainer page. Theme-safe: neutrals currentColor."""
    X0, X1, Y0, Y1 = 96, 884, 40, 328
    def sx(e): return X0 + (X1 - X0) * e / 250_000.0
    def sy(w): return Y1 - (Y1 - Y0) * w / 80.0

    def poly(vals):
        pts = [f"{sx(g):.1f},{sy(v):.1f}" for g, v in zip(GRID, vals) if np.isfinite(v)]
        return " ".join(pts)

    SVG_STYLE = [
        ("uniform",         "uniform",             "currentColor", .45, "none",   1.6),
        ("per",             "PER",                 "currentColor", .45, "2 3",    1.6),
        ("cce_base_bs",     "CCE shipped 0.86%",   "currentColor", .95, "none",   2.0),
        ("greedy_base_max", "greedy_base max",     "#8fc3b8",      1.0, "none",   1.8),
        ("greedy_base_wm",  "greedy_base wmean",   "#8fc3b8",      1.0, "5 3",    1.8),
        ("greedy_cov_max",  "greedy_cov max 16%",  "#0d7c6b",      1.0, "none",   2.6),
        ("greedy_cov_wm",   "greedy_cov wmean",    "#0d7c6b",      1.0, "5 3",    2.6),
    ]
    L = []
    L.append('<svg viewBox="0 0 920 470" role="img" aria-label="Win rate over training for '
             'JaxNav 8x8 fill 0.3. Completed baselines uniform, PER and shipped CCE reach '
             '60 to 69 percent by 250,000 episodes. The greedy coverage arms are still '
             'running and their lines stop early where the runs currently are.">')
    # y grid + labels
    for w in (0, 20, 40, 60, 80):
        L.append(f'<line x1="{X0}" y1="{sy(w):.1f}" x2="{X1}" y2="{sy(w):.1f}" '
                 f'stroke="currentColor" stroke-width="1" opacity="{.28 if w else .5}"/>')
        L.append(f'<text x="{X0-10}" y="{sy(w)+4:.1f}" text-anchor="end" '
                 f'font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor" '
                 f'opacity=".65">{w}%</text>')
    # x labels
    for e in (0, 62_500, 125_000, 187_500, 250_000):
        L.append(f'<text x="{sx(e):.1f}" y="{Y1+22:.0f}" text-anchor="middle" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10.5" fill="currentColor" '
                 f'opacity=".6">{e//1000}k</text>')
    # uniform IQR band
    u = curves.get("uniform")
    if u is not None:
        up = poly(u["hi"]); dn = poly(u["lo"])
        if up and dn:
            rev = " ".join(reversed(dn.split()))
            L.append(f'<polygon points="{up} {rev}" fill="currentColor" opacity=".10"/>')
    # epsilon marker
    L.append(f'<line x1="{sx(EPS_DECAY):.1f}" y1="{Y0}" x2="{sx(EPS_DECAY):.1f}" y2="{Y1}" '
             f'stroke="#a8341c" stroke-width="1.2" stroke-dasharray="4 4"/>')
    L.append(f'<text x="{sx(EPS_DECAY)+8:.1f}" y="{Y0+16:.0f}" '
             f'font-family="IBM Plex Mono, monospace" font-size="10" fill="#a8341c">'
             f'epsilon decay ends</text>')
    L.append(f'<text x="{sx(EPS_DECAY)+8:.1f}" y="{Y0+30:.0f}" '
             f'font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="#a8341c" '
             f'opacity=".85">left of here is exploration, not algorithm</text>')
    # curves
    for key, lab, colour, op, dash, lw in SVG_STYLE:
        c = curves.get(key)
        if c is None:
            continue
        p = poly(c["med"])
        if not p:
            continue
        da = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        L.append(f'<polyline points="{p}" fill="none" stroke="{colour}" stroke-width="{lw}" '
                 f'opacity="{op}"{da} stroke-linejoin="round"/>')
    # axes
    L.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="currentColor" stroke-width="1.2"/>')
    L.append(f'<text x="{X0}" y="{Y1+44:.0f}" font-family="IBM Plex Mono, monospace" '
             f'font-size="10.5" fill="currentColor" opacity=".6">episode</text>')
    # legend
    y = 384
    for i, (key, lab, colour, op, dash, lw) in enumerate(SVG_STYLE):
        c = curves.get(key)
        if c is None:
            continue
        col = X0 + (i % 3) * 268
        row = y + (i // 3) * 22
        da = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        L.append(f'<line x1="{col}" y1="{row-4}" x2="{col+26}" y2="{row-4}" stroke="{colour}" '
                 f'stroke-width="{lw}" opacity="{op}"{da}/>')
        st = "" if c["reach"] > 240_000 else "  · running"
        L.append(f'<text x="{col+34}" y="{row}" font-family="IBM Plex Sans, sans-serif" '
                 f'font-size="11.5" fill="currentColor" opacity=".8">{lab}  n={c["n"]}{st}</text>')
    L.append("</svg>")
    out = os.path.join(OUTDIR, "greedy_learning_curves.svg")
    open(out, "w").write("\n".join(L))
    print("wrote", out)
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    if "--norm" in sys.argv:
        cv = collect(_label_norm)
        for k, v in sorted(cv.items()):
            print(f"  {k:<16} n={v['n']:<3} reaches {v['reach']:>9,.0f} episodes")
        write_png(cv, NORM_STYLE,
                  "JaxNav 8x8 fill 0.3 — priority normalisation at 16% coverage (issue #16)\n"
                  "all three CCE arms identical except normalise_priorities; STILL RUNNING",
                  "norm_learning_curves.png")
        raise SystemExit(0)
    if "--capacity" in sys.argv:
        cv = collect(_label_capacity)
        for k, v in sorted(cv.items()):
            print(f"  {k:<16} n={v['n']:<3} reaches {v['reach']:>9,.0f} episodes")
        write_png(cv, CAP_STYLE,
                  "JaxNav 8x8 fill 0.3 — shrinking the buffer to buy coverage\n"
                  "every reduced-capacity arm collapses; 4x scoring does not rescue it",
                  "capacity_learning_curves.png")
        raise SystemExit(0)
    cv = collect()
    for k, v in sorted(cv.items()):
        print(f"  {k:<18} n={v['n']:<3} reaches {v['reach']:>9,.0f} episodes")
    write_png(cv)
    write_svg(cv)
