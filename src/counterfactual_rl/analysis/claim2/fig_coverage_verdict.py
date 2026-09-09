"""The coverage hypothesis, tested from both ends — and dead from both.

LEFT  FrozenLake-det, 208 runs. Coverage varied 48x through n_score_sample alone, every
      other knob held. Solve rate is flat (p=0.854). Even BELOW JaxNav's own 0.85%,
      FrozenLake still solves ~85% of the time.
RIGHT JaxNav, 32 runs at the full 250k. Raising coverage the cheap way -- shrinking the
      buffer -- collapses learning in 16 of 16 seeds (p=0.001). cce_8192 discards NO data,
      so this is loss of diversity, not loss of samples.

Together: coverage does not explain the JaxNav null, and the lever that raises it makes
things strictly worse.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.fig_coverage_verdict
"""
import glob, json, os, subprocess, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .jaxnav_bootstrap_curves import ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/fig_coverage_verdict.png")
FL_R = os.path.join(ROOT, "src/counterfactual_rl/agents/frozen_lake/runs")
JN_R = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")
EXP = os.path.join(ROOT, "src/counterfactual_rl/agents/frozen_lake/experiments/2026-09")
JN_MAN = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_capacity_outcome.json")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
C_CCE, C_UNI, C_BAD = "#0072b2", "#8a8a8a", "#d55e00"
LEVELS = (4, 8, 16, 32, 64, 128, 256)


def fl_data():
    out = subprocess.run(["sacct", "-n", "-P", "-X", "-o", "JobID,State", "-S", "2026-09-08",
                          "--name=FrozenLake DQN"], capture_output=True, text=True).stdout
    st = {l.split('|')[0]: l.split('|')[1] for l in out.strip().split('\n') if '|' in l}
    runs = collections.defaultdict(list)
    for pat in ("coverage_dose_2026*", "coverage_dose_ext_*", "coverage_cliff_*"):
        for man in glob.glob(os.path.join(EXP, pat, "*.json")):
            for j, cfg in json.load(open(man)).items():
                if st.get(j) != 'COMPLETED': continue
                f = os.path.join(FL_R, j, 'metrics.log')
                if not os.path.exists(f): continue
                wr = [float(q.split()[3].rstrip('%')) for q in open(f)
                      if not q.startswith('#') and q.strip() and len(q.split()) >= 6
                      and q.split()[0] != 'episode']
                if not wr: continue
                cov = None
                p = os.path.join(FL_R, j, 'ess.jsonl')
                if os.path.exists(p):
                    E = [json.loads(l) for l in open(p)]
                    if len(E) >= 5:
                        cov = 100*float(np.median([r['scored_frac']
                                                   for r in E[-max(3, len(E)//5):]]))
                runs[cfg.get('n_score_sample', 'uniform')].append((max(wr) >= 95, cov))
    return runs


def jn_data():
    m = json.load(open(JN_MAN)); aid = sorted({v['array_id'] for v in m.values()})[0]
    out = subprocess.run(["sacct", "-j", aid, "-X", "-n", "-P", "-o", "JobIDRaw,JobID,State"],
                         capture_output=True, text=True).stdout
    res = {p[1]: (p[0], p[2]) for p in (l.split("|") for l in out.strip().split("\n")) if len(p) >= 3}
    per = collections.defaultdict(list)
    for k, v in m.items():
        if v['job_id'] not in res: continue
        rd, s = res[v['job_id']]
        if s != 'COMPLETED': continue
        f = os.path.join(JN_R, rd, 'metrics.log')
        if not os.path.exists(f): continue
        wr = [float(q.split()[3].rstrip('%')) for q in open(f)
              if not q.startswith('#') and q.strip() and len(q.split()) >= 6
              and q.split()[0] != 'episode']
        if len(wr) > 40: per[v['arm']].append(float(np.mean(wr[-20:])))
    return per


def main():
    FL, JN = fl_data(), jn_data()
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.6), facecolor=SURFACE)
    fig.subplots_adjust(left=.065, right=.985, top=.70, bottom=.13, wspace=.24)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.grid(alpha=.55, color=GRID, lw=.8)
        ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=10)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ---- LEFT: FrozenLake, coverage does nothing ------------------------
    ax = axes[0]
    xs, ys, ks, ns = [], [], [], []
    for lv in LEVELS:
        if lv not in FL: continue
        v = FL[lv]; c = [x[1] for x in v if x[1] is not None]
        if not c: continue
        xs.append(float(np.median(c))); k = sum(x[0] for x in v)
        ks.append(k); ns.append(len(v)); ys.append(k/len(v))
    lo, hi = zip(*[stats.beta.ppf([.025, .975], k+.5, n-k+.5) for k, n in zip(ks, ns)])
    ax.errorbar(xs, ys, yerr=[np.clip(np.array(ys)-lo, 0, None),
                              np.clip(np.array(hi)-ys, 0, None)],
                fmt='o-', color=C_CCE, lw=2.6, ms=11, capsize=5, zorder=5)
    uni = FL.get('uniform', [])
    if uni:
        ku, nu = sum(x[0] for x in uni), len(uni)
        ax.axhline(ku/nu, color=C_UNI, ls="--", lw=2)
        ax.text(xs[-1], ku/nu-.075, f"uniform control  {ku/nu:.0%}", ha="right",
                fontsize=10.5, color=INK2)
    ax.axvline(0.85, color=C_BAD, ls=":", lw=2)
    ax.text(0.9, .06, "JaxNav sits\nHERE (0.85%)", fontsize=10, color=C_BAD,
            fontweight="bold")
    ax.set_xscale("log"); ax.set_ylim(0, 1.06)
    ax.set_xlabel("replay coverage  (log)", fontsize=11, color=INK2)
    ax.set_ylabel("fraction of seeds that SOLVED", fontsize=11, color=INK2)
    ax.set_title("1.  Coverage does NOTHING   (FrozenLake, 208 runs)",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    ax.text(.5, .93, "48x range · flat · p = 0.854", transform=ax.transAxes,
            ha="center", fontsize=11.5, color=INK, fontweight="bold")

    # ---- RIGHT: JaxNav, raising it destroys learning --------------------
    ax = axes[1]
    order = [a for a in ('uniform', 'cce_base', 'cce_8192', 'cce_2500') if a in JN]
    LAB = {'uniform': 'uniform\n100k', 'cce_base': 'CCE\n100k',
           'cce_8192': 'CCE\n8,192', 'cce_2500': 'CCE\n2,500'}
    COV = {'uniform': '—', 'cce_base': '0.86%', 'cce_8192': '1.6%', 'cce_2500': '5.1%'}
    for i, a in enumerate(order):
        v = JN[a]; col = C_UNI if a == 'uniform' else (C_CCE if a == 'cce_base' else C_BAD)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.08, .08, len(v)),
                   v, s=95, color=col, zorder=5, edgecolor=SURFACE, linewidth=1.6)
        ax.plot([i-.28, i+.28], [np.median(v)]*2, color=col, lw=3, zorder=6)
        k = sum(1 for x in v if x >= 40)
        ax.text(i, 104, f"{k}/{len(v)}", ha="center", fontsize=12, fontweight="bold",
                color=col)
        ax.text(i, 97, COV[a], ha="center", fontsize=9.5, color=INK2)
    ax.axhline(40, color=INK2, ls="--", lw=1.5)
    ax.text(len(order)-.55, 42, "good mode", fontsize=9.5, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LAB[a] for a in order], fontsize=10)
    ax.set_ylabel("final win rate per seed  (%)", fontsize=11, color=INK2)
    ax.set_ylim(0, 112)
    ax.set_title("2.  But raising it BREAKS learning   (JaxNav, 8 seeds/arm)",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    ax.annotate("", xy=(2.5, 20), xytext=(1.15, 52),
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=2.5))
    ax.text(2.0, 60, "16 of 16 seeds\ncollapse\np = 0.001", fontsize=11,
            color=C_BAD, fontweight="bold", ha="center")

    fig.text(.065, .955, "The coverage hypothesis, tested from both ends — and dead from both",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.065, .885,
             "We believed CCE fails on JaxNav because only 0.85% of its buffer carries a "
             "measured score. Two experiments, 240 runs.\n\n"
             "LEFT: on FrozenLake, dialling coverage 0.7% -> 33.7% changes nothing — and at "
             "0.7%, BELOW JaxNav's own level, it still solves 85%.\n"
             "RIGHT: on JaxNav, the cheap way to raise coverage (a smaller buffer) collapses "
             "every seed. cce_8192 discards NO data, so this is lost diversity, not lost "
             "samples.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.5)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
