"""FrozenLake vs JaxNav: what each replay buffer actually contains (issue #18).

Jeremy's 09/03 action item 2. Until now the JaxNav column was measured and the
FrozenLake column was arithmetic on config values -- the same arithmetic that was
wrong by ~2x on JaxNav. Both columns are now measured.

Safe to run mid-sweep: uses whatever runs have written ess.jsonl / buffer snapshots,
and prints how many of each it found.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.fl_vs_jaxnav_buffer
"""
import glob, json, os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .jaxnav_bootstrap_curves import ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/fig_fl_vs_jaxnav_buffer.png")
FL_R = os.path.join(ROOT, "src/counterfactual_rl/agents/frozen_lake/runs")
JN_R = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
JN_MAN = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_instrument.json")
C_FL, C_JN = "#009e73", "#d55e00"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"


def fl_runs():
    man = glob.glob(os.path.join(
        ROOT, "src/counterfactual_rl/agents/frozen_lake/experiments/2026-09/"
              "buffer_reference_*/*.json"))
    if not man:
        return {}
    m = json.load(open(man[0]))
    return {j: c for j, c in m.items()
            if c['algorithm'] == 'consequence-dqn'
            and os.path.exists(os.path.join(FL_R, j, 'ess.jsonl'))}


def jn_runs():
    m = json.load(open(JN_MAN))
    out = subprocess.run(["sacct", "-j", "274678", "-X", "-n", "-P",
                          "-o", "JobIDRaw,JobID,State"], capture_output=True, text=True).stdout
    res = {p[1]: p[0] for p in (l.split("|") for l in out.strip().split("\n")) if len(p) >= 3}
    return {res[r['job_id']]: r for r in m.values()
            if r['arm'].startswith('cce') and r['job_id'] in res
            and os.path.exists(os.path.join(JN_R, res[r['job_id']], 'ess.jsonl'))}


def zero_traj(run_dir):
    """(episode, zero-rate-of-measured) from each buffer snapshot."""
    eps, zr = [], []
    for sp in sorted(glob.glob(os.path.join(run_dir, 'buffer', '*.npz'))):
        z = np.load(sp); cs, sa = z['consequence_scores'], z['scored_at']
        m = sa != -1
        if m.sum() < 10:
            continue
        eps.append(int(os.path.basename(sp).split('_')[1].split('.')[0]))
        zr.append(100 * float((cs[m] <= 0).mean()))
    return np.array(eps), np.array(zr)


def main():
    FL, JN = fl_runs(), jn_runs()
    print(f"FrozenLake CCE runs with ess.jsonl: {len(FL)}   JaxNav: {len(JN)}")
    if not FL or not JN:
        print("not enough data yet"); return

    def last(rd):
        return json.loads(open(os.path.join(rd, 'ess.jsonl')).readlines()[-1])

    fl_last = [last(os.path.join(FL_R, j)) for j in FL]
    jn_last = [last(os.path.join(JN_R, j)) for j in JN]

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=.045, right=.99, top=.62, bottom=.14, wspace=.32)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.grid(alpha=.55, color=GRID, lw=.8)
        ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    def pair(ax, fl_v, jn_v, ylab, title, log=False, fmt="{:.2f}"):
        for i, (v, c, lab) in enumerate([(fl_v, C_FL, "FrozenLake-det\nCCE WINS"),
                                         (jn_v, C_JN, "JaxNav\nCCE NULL")]):
            ax.bar(i, np.median(v), .56, color=c, alpha=.92)
            ax.scatter([i]*len(v), v, s=34, color=INK, zorder=5,
                       edgecolor=SURFACE, linewidth=1.1)
            med = np.median(v)
            # Label ABOVE the bar with real headroom. The previous version placed it
            # at the bar top and it collided with the bar and the seed dots.
            # Offset the label sideways as well as up: on the log panels the seed
            # dots sit directly above the bar top and collided with a centred label.
            y = med * 2.1 if log else med + max(np.median(fl_v), np.median(jn_v)) * .09
            ax.text(i + (.34 if log else 0), y, fmt.format(med),
                    ha="left" if log else "center", fontsize=11.5,
                    fontweight="bold", color=c)
        if log: ax.set_yscale("log")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["FrozenLake-det\nCCE WINS",
                                                   "JaxNav\nCCE NULL"], fontsize=9)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, loc="left", pad=9)
        if not log:
            ax.set_ylim(0, max(max(fl_v), max(jn_v)) * 1.30)

    # 1. coverage OVER TRAINING, not a snapshot -- the earlier version showed only the
    # final value, which asserted rather than showed that coverage is stationary.
    ax = axes[0]
    def cov_traj(runs, R):
        for j in runs:
            E = [json.loads(l) for l in open(os.path.join(R, j, 'ess.jsonl'))]
            e = np.array([r['episode'] for r in E], float)
            c = np.array([100*r['scored_frac'] for r in E], float)
            if len(e) > 2:
                yield e/e.max(), c
    for e, c in cov_traj(FL, FL_R):
        ax.plot(e, c, lw=1.9, color=C_FL, alpha=.85)
    for e, c in cov_traj(JN, JN_R):
        ax.plot(e, c, lw=1.9, color=C_JN, alpha=.85)
    ax.set_yscale("log"); ax.set_ylim(.4, 60)
    ax.set_xlabel("fraction of training completed", fontsize=10, color=INK2)
    ax.set_ylabel("% of buffer with a MEASURED score", fontsize=10, color=INK2)
    ax.set_title("1.  Replay coverage over training", fontsize=12, fontweight="bold",
                 color=INK, loc="left", pad=9)
    fl_c = [100*r['scored_frac'] for r in fl_last]; jn_c = [100*r['scored_frac'] for r in jn_last]
    ax.text(.52, np.median(fl_c)*1.55, f"FrozenLake  {np.median(fl_c):.1f}%",
            fontsize=10.5, color=C_FL, fontweight="bold")
    ax.text(.52, np.median(jn_c)*1.55, f"JaxNav  {np.median(jn_c):.2f}%",
            fontsize=10.5, color=C_JN, fontweight="bold")
    ax.text(.03, 33, f"{np.median(fl_c)/np.median(jn_c):.0f}x", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(.03, .62, "both FLAT once the buffer fills —\nthe early bump is the fill phase,\n"
                      "not a trend. Coverage is set by\nconfig, not by training time.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")

    # 2. zero-rate trajectory
    ax = axes[1]
    # Split FL by whether the seed EVER solved. Without this the panel compares FL
    # seeds that never reached the goal against JaxNav seeds that learned, and it
    # looks like FL has the worse zero-rate -- which is the matched-competence trap,
    # not a finding. A seed that never reaches the goal has every rollout return 0
    # BY CONSTRUCTION, in either environment.
    n_solved = 0
    for j in FL:
        e, z = zero_traj(os.path.join(FL_R, j))
        if not len(e):
            continue
        wr = []
        for line in open(os.path.join(FL_R, j, 'metrics.log')):
            if line.startswith('#') or not line.strip():
                continue
            q = line.split()
            if len(q) >= 6 and q[0] != 'episode':
                wr.append(float(q[3].rstrip('%')))
        solved = bool(wr) and max(wr) >= 95
        n_solved += solved
        ax.plot(e/e.max(), z, lw=2.2 if solved else 1.2, color=C_FL,
                alpha=.95 if solved else .40,
                ls='-' if solved else (0, (3, 2)))
    for j in JN:
        e, z = zero_traj(os.path.join(JN_R, j))
        if len(e): ax.plot(e/e.max(), z, lw=1.9, color=C_JN, alpha=.85)
    ax.set_xlabel("fraction of training completed", fontsize=10, color=INK2)
    ax.set_ylabel("% of MEASURED scores that are exactly 0", fontsize=10, color=INK2)
    ax.set_title("2.  The zero-rate is a TRAJECTORY", fontsize=12, fontweight="bold",
                 color=INK, loc="left", pad=9)
    ax.set_ylim(-4, 108)
    ax.plot([], [], color=C_FL, lw=2.2, label="FrozenLake — seed SOLVED it")
    ax.plot([], [], color=C_FL, lw=1.2, ls=(0, (3, 2)), alpha=.5,
            label="FrozenLake — never solved")
    ax.plot([], [], color=C_JN, lw=2, label="JaxNav (all learned)")
    ax.legend(fontsize=8.5, frameon=False, loc="center left", bbox_to_anchor=(0, .30))

    # 3. staleness
    fl_s = [r['staleness_p50'] for r in fl_last if r.get('staleness_p50')]
    jn_s = [r['staleness_p50'] for r in jn_last if r.get('staleness_p50')]
    pair(axes[2], fl_s, jn_s, "age of the score, Q-updates (p50)",
         "3.  Staleness — FL is WORSE", fmt="{:,.0f}")

    # 4. usable-signal funnel
    ax = axes[3]
    def usable(runs, R):
        out = []
        for j in runs:
            sn = sorted(glob.glob(os.path.join(R, j, 'buffer', '*.npz')))
            if not sn: continue
            z = np.load(sn[-1]); cs, sa = z['consequence_scores'], z['scored_at']
            m = sa != -1
            if m.sum() < 10: continue
            out.append(100 * float((m & (cs > 0)).mean()))
        return out
    fu, ju = usable(FL, FL_R), usable(JN, JN_R)
    if fu and ju:
        pair(ax, fu, ju, "% of buffer with a NON-ZERO measured score",
             "4.  Usable signal (coverage x non-zero)", log=True, fmt="{:.2f}%")
        ax.set_ylim(.02, 60)
        ax.text(.5, 8, f"{np.median(fu)/np.median(ju):.0f}x", ha="center",
                fontsize=15, fontweight="bold", color=INK)

    fig.text(.045, .965, "What each replay buffer actually contains",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.045, .905,
             f"FrozenLake-det 8x8 ({len(FL)} CCE runs, array 274850-862) vs JaxNav 8x8 "
             f"fill 0.3 ({len(JN)} CCE runs, array 274678).  Panels 1-2: one line per seed. "
             f"Panels 3-4: median, dots are seeds.\n"
             "Until today the FrozenLake column was ARITHMETIC on config values. Both are now "
             "measured. CAVEAT ON PANELS 2 AND 4: only 4 of 12 FL seeds solved the map, and a "
             "seed that never\nreaches the goal has 100% zero scores BY CONSTRUCTION — so the "
             "green curves at 100%, and FL's low usable signal, are mostly about seeds that "
             "never learned.",
             fontsize=10, color=INK2, va="top", linespacing=1.6)
    fig.savefig(OUT, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
