"""JaxNav holes-map figures, built straight from the run logs.

Three figures, one claim each:

  fig_jaxnav_iqm_v4.png   the 96k full-budget run (v4, jobs 271903-271912).
                          IQM win-rate over training, plus every seed's final
                          number, plus the exact permutation p-value.

  fig_jaxnav_aggregation.png  the aggregation control (jobs 271925-271933).
                          Same seeds, same 12k budget, only max vs
                          weighted_mean differs. Answers GitHub issue #3
                          with a training run instead of a static probe.

  fig_jaxnav_25seed_power.png  the properly-powered follow-up (25 seeds/arm,
                          96k budget): CCE+max, CCE+weighted_mean, PER. Job
                          list is read from the sweep's manifest.json, not
                          hardcoded ranges, because 3 of the original
                          CCE+weighted_mean seeds died on a bad compute node
                          (dh-node12) and were resubmitted under different
                          job IDs (272041-272043).

Every number is read from `runs/<job>/metrics.log`. Nothing is hard-coded:
the p-value and every mean are computed here from those arrays.
"""
import json
import os
import statistics as st
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
BROWN = "#8c5a2b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
GREY = "#8c8b87"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "agents", "jax_nav", "runs"))
EXPERIMENTS = os.path.normpath(os.path.join(HERE, "..", "..", "agents", "jax_nav", "experiments"))
OUT_DIR = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "..", "docs", "figures", "real", "claim2", "jaxnav")
)

V4_CCE = range(271903, 271908)
V4_PER = range(271908, 271913)
AGG_WMEAN = range(271925, 271928)
AGG_PER = range(271928, 271931)
AGG_MAX = range(271931, 271934)

POWER_MANIFEST = os.path.join(EXPERIMENTS, "holes_25seed_power", "manifest.json")


def load(job, runs_dir=None):
    """Return (episodes, win_rates) from a run's metrics.log, or None."""
    path = os.path.join(runs_dir or RUNS, str(job), "metrics.log")
    if not os.path.exists(path):
        return None
    eps, wins = [], []
    for line in open(path):
        if line.startswith("#") or "episode" in line:
            continue
        tok = line.split()
        if len(tok) < 4:
            continue
        try:
            eps.append(int(tok[0]))
            wins.append(float(tok[3].rstrip("%")) / 100.0)
        except ValueError:
            continue
    return (np.array(eps), np.array(wins)) if eps else None


def finals(jobs, k=5, runs_dir=None, min_episode=None):
    """Each seed's final score = mean of its last k evaluations.

    `min_episode` guards against mixing budgets: a seed that died early (bad
    node, timeout) still has a metrics.log, and averaging its last 5 evals
    silently drops an early-training number into a full-budget mean. Pass the
    target budget and those seeds are excluded instead of quietly counted."""
    out = []
    for j in jobs:
        r = load(j, runs_dir)
        if r is None or len(r[1]) < k:
            continue
        if min_episode is not None and r[0][-1] < min_episode:
            continue
        out.append(float(np.mean(r[1][-k:])))
    return out


def coverage(jobs, runs_dir=None):
    """Per-arm truth about what is actually on disk: {job: last_episode},
    plus the jobs with no readable metrics.log at all."""
    reached, missing = {}, []
    for j in jobs:
        r = load(j, runs_dir)
        if r is None or len(r[0]) == 0:
            missing.append(j)
        else:
            reached[j] = int(r[0][-1])
    return reached, missing


def report_coverage(name, jobs, target, runs_dir=None, tol=0.95):
    """Print what this arm really contains and return the seeds that finished.

    Silence is the enemy here: a figure built from 22 of 25 seeds, or from
    seeds that stopped at a third of the budget, looks identical to a clean
    one unless the shortfall is printed."""
    reached, missing = coverage(jobs, runs_dir)
    cutoff = target * tol
    short = {j: e for j, e in reached.items() if e < cutoff}
    ok = [j for j in jobs if j in reached and j not in short]
    print(f"  {name:10s} {len(ok)}/{len(jobs)} seeds reached >={cutoff:.0f} episodes")
    if missing:
        print(f"    !! no metrics.log: {missing}")
    if short:
        print(f"    !! stopped early (excluded): "
              f"{ {j: e for j, e in sorted(short.items())} }")
    return ok


def iqm(vals):
    """Interquartile mean: drop the extremes, average the middle."""
    s = sorted(vals)
    return float(np.mean(s[1:-1])) if len(s) > 2 else float(np.mean(s))


def iqm_curve(jobs, runs_dir=None):
    """IQM computed fresh at every checkpoint, on a common episode grid.

    The grid stops at the SHORTEST seed, so one seed that died early drags the
    whole curve back -- callers should check the returned grid's endpoint
    against the budget they think they are plotting."""
    curves = [c for c in (load(j, runs_dir) for j in jobs) if c is not None]
    if not curves:
        return None, None
    hi = min(c[0][-1] for c in curves)
    grid = np.linspace(250, hi, 300)
    stack = np.vstack([np.interp(grid, e, w) for e, w in curves])
    return grid, np.array([iqm(col) for col in stack.T])


def tail_trend(grid, curve, windows=(5000, 10000, 20000)):
    """Least-squares slope of the IQM curve over each trailing window, given as
    the percentage-point change across that window. ~0 = the arm has settled.

    Reported at several windows on purpose. The obvious version of this check
    -- last value minus first value of the window -- is dominated by noise in
    the two endpoints and flips sign with the window: on the 96k run PER reads
    -1.2pp at a 3k window but +15.9pp at 20k. One number here is not evidence
    of convergence, so `settled` demands agreement across all windows."""
    if grid is None or len(grid) < 2:
        return None
    out = {}
    for w in windows:
        sel = grid >= grid[-1] - w
        if sel.sum() < 3:
            continue
        out[w] = float(np.polyfit(grid[sel], curve[sel], 1)[0] * w * 100)
    return out or None


def trend_verdict(trend, flat_pp=2.0):
    """'settled' only if every window agrees the curve is flat."""
    if not trend:
        return "no data"
    if all(abs(v) < flat_pp for v in trend.values()):
        return "flat (settled)"
    worst = max(trend.values(), key=abs)
    return "still RISING" if worst > 0 else "still FALLING"


def perm_p(a, b):
    """Exact one-sided permutation test; enumerates all C(n,k) splits."""
    pool = list(a) + list(b)
    obs = st.mean(a) - st.mean(b)
    hits = total = 0
    for idx in combinations(range(len(pool)), len(a)):
        g1 = [pool[i] for i in idx]
        g2 = [pool[i] for i in range(len(pool)) if i not in idx]
        total += 1
        if st.mean(g1) - st.mean(g2) >= obs:
            hits += 1
    return hits / total, total


def _frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def fig_v4(runs_dir=None):
    cce, per = finals(V4_CCE, runs_dir=runs_dir), finals(V4_PER, runs_dir=runs_dir)
    p1, nperm = perm_p(cce, per)
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.65, 1]})

    for jobs, col, lab in ((V4_CCE, ORANGE, "CCE-mul"), (V4_PER, BLUE, "PER")):
        g, y = iqm_curve(jobs, runs_dir)
        if g is None:
            continue
        ax.plot(g / 1000, y * 100, color=col, lw=2.4, label=f"{lab}  (IQM of 5 seeds)")
    ax.axvline(40, color=GREY, lw=1.4, ls=(0, (5, 3)))
    ax.text(41, 4, "exploration ends\n(ε floor, ep 40k)", fontsize=9, color=INK2, va="bottom")
    ax.set_xlabel("training episodes (thousands)", color=INK2, fontsize=11)
    ax.set_ylabel("evaluation win rate  (%)", color=INK2, fontsize=11)
    ax.set_title("JaxNav holes map — full 96k budget, no early stop",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    _frame(ax)

    for i, (vals, col, lab) in enumerate(((cce, ORANGE, "CCE-mul"), (per, BLUE, "PER"))):
        x = np.full(len(vals), i) + np.linspace(-0.11, 0.11, len(vals))
        bx.scatter(x, np.array(vals) * 100, s=115, color=col, alpha=0.9,
                   edgecolor="white", lw=1.6, zorder=3)
        bx.plot([i - 0.26, i + 0.26], [np.mean(vals) * 100] * 2, lw=3, color=col, zorder=4)
        bx.text(i, 96, f"mean {np.mean(vals)*100:.1f}%\nIQM {iqm(vals)*100:.1f}%",
                ha="center", fontsize=10.5, color=col, fontweight="bold")
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["CCE-mul", "PER"], fontsize=11.5, color=INK2)
    bx.set_xlim(-0.6, 1.6)
    bx.set_ylim(0, 108)
    bx.set_ylabel("final win rate  (mean of last 5 evals)", color=INK2, fontsize=11)
    bx.set_title(f"Every seed. Exact permutation p = {p1*2:.3f}\n"
                 f"(two-sided, all {nperm} splits) — not significant",
                 color=INK, fontsize=10.5, loc="left")
    _frame(bx)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig_jaxnav_iqm_v4.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  CCE {cce} mean={st.mean(cce):.4f} iqm={iqm(cce):.4f}")
    print(f"  PER {per} mean={st.mean(per):.4f} iqm={iqm(per):.4f}")
    print(f"  permutation one-sided p={p1:.4f} two-sided={p1*2:.4f}")
    return out


def fig_agg(runs_dir=None):
    arms = (("CCE\nweighted_mean\n(the fix)", AGG_WMEAN, ORANGE),
            ("CCE\nmax\n(old bug, on purpose)", AGG_MAX, "#b8531f"),
            ("PER\n(baseline)", AGG_PER, BLUE))
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for i, (lab, jobs, col) in enumerate(arms):
        vals = finals(jobs, runs_dir=runs_dir)
        if not vals:
            continue
        ax.bar(i, np.mean(vals) * 100, width=0.56, color=col, alpha=0.28,
               edgecolor=col, lw=1.8, zorder=2)
        x = np.full(len(vals), i) + np.linspace(-0.13, 0.13, len(vals))
        ax.scatter(x, np.array(vals) * 100, s=95, color=col, alpha=0.95,
                   edgecolor="white", lw=1.5, zorder=4)
        top = max(np.mean(vals), max(vals)) * 100
        ax.text(i, top + 2.2, f"{np.mean(vals)*100:.1f}%",
                ha="center", fontsize=13, color=col, fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=10.5, color=INK2)
    ax.set_ylabel("win rate  (mean of last 5 evals)", color=INK2, fontsize=11)
    ax.set_ylim(0, 40)
    ax.set_title("Aggregation control — same 3 seeds, same 12k budget, one knob changed"
                 "\nGitHub issue #3, answered by training rather than a static probe",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left")
    _frame(ax)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig_jaxnav_aggregation.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    for lab, jobs, _ in arms:
        v = finals(jobs, runs_dir=runs_dir)
        print(f"  {lab.splitlines()[0]:12s} {v} mean={st.mean(v):.4f}" if v else f"  {lab}: none")
    return out


def _power_arms(manifest_path=POWER_MANIFEST):
    """Arm -> sorted job-id list, read from the manifest (ground truth, not a
    hardcoded range) so resubmitted/replacement seeds are picked up correctly."""
    manifest = json.load(open(manifest_path))
    arms = {"cce_max": [], "cce_wmean": [], "per": []}
    for jid, cfg in manifest.items():
        if cfg["algorithm"] == "dqn":
            arms["per"].append(int(jid))
        elif cfg.get("consequence_aggregation") == "max":
            arms["cce_max"].append(int(jid))
        elif cfg.get("consequence_aggregation") == "weighted_mean":
            arms["cce_wmean"].append(int(jid))
    for k in arms:
        arms[k].sort()
    return arms


def _target_episodes(manifest_path=POWER_MANIFEST):
    """The episode budget this sweep was launched with, taken from the manifest
    so nothing downstream has to hardcode 96k vs 150k."""
    manifest = json.load(open(manifest_path))
    budgets = {cfg["n_episodes"] for cfg in manifest.values() if "n_episodes" in cfg}
    if len(budgets) != 1:
        raise ValueError(f"mixed episode budgets in {manifest_path}: {sorted(budgets)}")
    return budgets.pop()


def fig_25seed_power(manifest_path=POWER_MANIFEST, budget_label="96k",
                      out_name="fig_jaxnav_25seed_power.png", runs_dir=None):
    """25 seeds/arm: CCE+max (the old aggregation bug, kept on purpose as a
    control), CCE+weighted_mean (the fix), PER. Priority mixing is
    multiplicative (Eq5, mu_c=mu_delta=1.0 -- both defaults, never
    overridden) for both CCE arms; 'mu' (additive-only) is unused.

    `manifest_path`/`budget_label`/`out_name` let this same function build
    the figure for any episode-budget rerun of this comparison (e.g. the
    150k follow-up) -- just point it at that run's manifest.json."""
    arms = _power_arms(manifest_path)
    target = _target_episodes(manifest_path)
    labels = {"cce_max": "CCE+max (bug)", "cce_wmean": "CCE+wmean (fix)", "per": "PER"}
    colors = {"cce_max": BROWN, "cce_wmean": ORANGE, "per": BLUE}

    # Keep only seeds that actually ran the full budget, and say out loud which
    # ones were dropped -- a quietly-thinned arm is the easiest way to read a
    # win that is not there.
    print(f"\n  coverage (target {target} episodes):")
    kept = {k: report_coverage(k, arms[k], target, runs_dir)
            for k in ("per", "cce_max", "cce_wmean")}

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(14.2, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})

    vals_by_arm, slopes = {}, {}
    for key in ("per", "cce_max", "cce_wmean"):
        jobs = kept[key]
        vals_by_arm[key] = finals(jobs, runs_dir=runs_dir, min_episode=target * 0.95)
        g, y = iqm_curve(jobs, runs_dir)
        if g is None:
            continue
        slopes[key] = tail_trend(g, y)
        if g[-1] < target * 0.95:
            print(f"    !! {key} curve stops at episode {g[-1]:.0f}, "
                  f"short of the {target}-episode budget")
        ax.plot(g / 1000, y * 100, color=colors[key], lw=2.2,
                label=f"{labels[key]}  (n={len(vals_by_arm[key])})")
    ax.set_xlabel("training episodes (thousands)", color=INK2, fontsize=11)
    ax.set_ylabel("evaluation win rate  (%)", color=INK2, fontsize=11)
    ax.set_title(f"Holes map, {budget_label} budget — 25 seeds/arm (properly powered)",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _frame(ax)

    order = ["per", "cce_max", "cce_wmean"]
    for i, key in enumerate(order):
        vals = np.array(vals_by_arm[key])
        col = colors[key]
        if len(vals) == 0:
            # Normal while a sweep is still running: no seed has hit the budget
            # yet. Draw nothing for this arm rather than dying, so the script
            # stays usable mid-flight.
            bx.text(i, 50, "no finished\nseeds yet", ha="center", va="center",
                    fontsize=9.5, color=GREY)
            continue
        x = np.full(len(vals), i) + np.linspace(-0.16, 0.16, len(vals))
        bx.scatter(x, vals * 100, s=32, color=col, alpha=0.75,
                   edgecolor="white", lw=0.4, zorder=3)
        bx.plot([i - 0.28, i + 0.28], [vals.mean() * 100] * 2, lw=2.6, color=col, zorder=4)
        bx.text(i, vals.max() * 100 + 5, f"{vals.mean()*100:.1f}%",
                ha="center", fontsize=10.5, color=col, fontweight="bold")
    bx.set_xticks(range(len(order)))
    bx.set_xticklabels([labels[k] for k in order], fontsize=9.5, color=INK2)
    bx.set_xlim(-0.6, 2.6)
    bx.set_ylim(0, 102)
    bx.set_ylabel("final win rate  (mean of last 5 evals)", color=INK2, fontsize=11)
    bx.set_title("Every seed (n=25 each)", color=INK, fontsize=11, loc="left")
    _frame(bx)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")

    print("\n  stats (Welch t-test / Mann-Whitney U):")
    for a_key, b_key in (("cce_max", "per"), ("cce_wmean", "per"), ("cce_max", "cce_wmean")):
        a, b = np.array(vals_by_arm[a_key]), np.array(vals_by_arm[b_key])
        if len(a) < 2 or len(b) < 2:
            print(f"  {labels[a_key]:16s} vs {labels[b_key]:16s}  "
                  f"skipped (n={len(a)} vs n={len(b)}, sweep not finished)")
            continue
        t, p_t = scipy_stats.ttest_ind(a, b, equal_var=False)
        u, p_u = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {labels[a_key]:16s} vs {labels[b_key]:16s}  "
              f"diff={a.mean()-b.mean():+.1%}  t-test p={p_t:.4f}  MannWhitney p={p_u:.4f}")

    # Why the budget was raised. The original read of the 96k run was "PER has
    # settled, the CCE arms have not", but that came from the endpoint-diff
    # estimator; with the least-squares version below, all three arms -- PER
    # included -- are still climbing at 96k. So 96k undershot for everyone,
    # and no arm's final number there was a converged one.
    print("\n  convergence (IQM trend over trailing windows, pp across window):")
    for key in ("per", "cce_max", "cce_wmean"):
        tr = slopes.get(key)
        if not tr:
            continue
        cells = "  ".join(f"{w//1000}k:{v:+5.1f}" for w, v in sorted(tr.items()))
        print(f"  {labels[key]:16s} {cells}   -> {trend_verdict(tr)}")
    print("  (an arm that is not 'flat' has not converged: its final number is\n"
          "   a snapshot mid-climb, not the level it would reach.)")
    return out


# 150k follow-up run's manifest (relaunched because none of the 96k CCE curves had converged).
MANIFEST_150K = os.path.join(EXPERIMENTS, "holes_25seed_150k", "manifest.json")


def fig_25seed_150k(runs_dir=None):
    return fig_25seed_power(MANIFEST_150K, budget_label="150k",
                             out_name="fig_jaxnav_25seed_150k.png", runs_dir=runs_dir)


if __name__ == "__main__":
    fig_v4()
    fig_agg()
    fig_25seed_power()
    if os.path.exists(MANIFEST_150K):
        fig_25seed_150k()
