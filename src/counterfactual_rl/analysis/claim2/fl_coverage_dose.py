"""FrozenLake coverage dose-response: does coverage CAUSE the win? (issue #22)

Everything before this was correlational across two environments -- FrozenLake 18.9%
coverage / CCE wins vs JaxNav 0.85% / CCE null -- and those differ on three axes at once.
Here coverage is a dial in ONE environment that has a positive control, varied only
through n_score_sample.

ONLY COMPLETED RUNS ARE USED. This is not defensive boilerplate: on 2026-09-08 twelve of
36 pilot runs were SIGKILLed on dh-node12 (a node the lab notebook has flagged since
2026-08-13, which this env's sbatch never excluded), and a first pass that keyed off "has
a metrics.log" silently counted every crashed run as "never solved" -- biasing the solve
rate DOWN, unevenly across levels, and producing a trend that was partly an artifact of
where jobs happened to land. Job state is now checked, and the count of excluded runs is
printed so a silent shortfall is visible.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.fl_coverage_dose
"""
import glob, json, os, subprocess, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .jaxnav_bootstrap_curves import ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/fig_fl_coverage_dose.png")
R = os.path.join(ROOT, "src/counterfactual_rl/agents/frozen_lake/runs")
EXP = os.path.join(ROOT, "src/counterfactual_rl/agents/frozen_lake/experiments/2026-09")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdad5", "#fcfcfb"
C_CCE, C_UNI = "#0072b2", "#8a8a8a"
LEVELS = (16, 32, 64, 128, 256)


def job_states():
    out = subprocess.run(["sacct", "-n", "-P", "-X", "-o", "JobID,State",
                          "-S", "2026-09-08", "--name=FrozenLake DQN"],
                         capture_output=True, text=True).stdout
    return {l.split('|')[0]: l.split('|')[1] for l in out.strip().split('\n') if '|' in l}


def load():
    st = job_states()
    runs, skipped = collections.defaultdict(list), collections.Counter()
    for pat in ("coverage_dose_2026*", "coverage_dose_ext_*"):
        for man in glob.glob(os.path.join(EXP, pat, "*.json")):
            for j, cfg in json.load(open(man)).items():
                lvl = cfg.get('n_score_sample', 'uniform')
                if st.get(j) != 'COMPLETED':
                    skipped[(lvl, st.get(j, 'missing'))] += 1
                    continue
                f = os.path.join(R, j, 'metrics.log')
                if not os.path.exists(f):
                    skipped[(lvl, 'no log')] += 1
                    continue
                wr = []
                for line in open(f):
                    if line.startswith('#') or not line.strip():
                        continue
                    q = line.split()
                    if len(q) >= 6 and q[0] != 'episode':
                        wr.append(float(q[3].rstrip('%')))
                if not wr:
                    skipped[(lvl, 'empty log')] += 1
                    continue
                cov = None
                p = os.path.join(R, j, 'ess.jsonl')
                if os.path.exists(p):
                    E = [json.loads(l) for l in open(p)]
                    if len(E) >= 5:
                        cov = 100*float(np.median([r['scored_frac']
                                                   for r in E[-max(3, len(E)//5):]]))
                runs[lvl].append((max(wr) >= 95, float(np.mean(wr[-20:])), cov))
    return runs, skipped


def main():
    runs, skipped = load()
    if not runs:
        print("no completed runs yet"); return
    print("EXCLUDED (not COMPLETED):", dict(skipped) if skipped else "none")
    print(f"\n{'n_score_sample':>14} {'n':>4} {'coverage':>10} {'solve rate':>12} {'95% CI':>16}")
    xs, ys, ns, ks, covs = [], [], [], [], []
    for lvl in LEVELS:
        if lvl not in runs: continue
        v = runs[lvl]; k = sum(x[0] for x in v); n = len(v)
        c = [x[2] for x in v if x[2] is not None]
        cov = float(np.median(c)) if c else float('nan')
        lo, hi = stats.beta.ppf([.025, .975], k+.5, n-k+.5)
        print(f"{lvl:>14} {n:>4} {cov:>9.1f}% {k/n:>11.0%} "
              f"{f'[{lo:.2f}, {hi:.2f}]':>16}")
        xs.append(cov); ys.append(k/n); ns.append(n); ks.append(k); covs.append(c)
    uni = runs.get('uniform', [])
    if uni:
        ku, nu = sum(x[0] for x in uni), len(uni)
        print(f"{'uniform':>14} {nu:>4} {'n/a':>10} {ku/nu:>11.0%}")

    # trend test over individual runs, not over level means
    allx, ally = [], []
    for lvl, cov in zip([l for l in LEVELS if l in runs], xs):
        for solved, _, _ in runs[lvl]:
            allx.append(np.log(cov)); ally.append(int(solved))
    if len(set(allx)) >= 3:
        r, p = stats.pearsonr(allx, ally)
        print(f"\n  trend over {len(allx)} runs: r = {r:+.3f}, p = {p:.4f}")
        if len(xs) >= 2:
            a = runs[LEVELS[0]]; b = runs[[l for l in LEVELS if l in runs][-1]]
            tab = [[sum(x[0] for x in b), len(b)-sum(x[0] for x in b)],
                   [sum(x[0] for x in a), len(a)-sum(x[0] for x in a)]]
            print(f"  endpoints Fisher exact: p = {stats.fisher_exact(tab)[1]:.4f}")

    fig, ax = plt.subplots(figsize=(9.5, 6.4), facecolor=SURFACE)
    fig.subplots_adjust(left=.10, right=.97, top=.74, bottom=.12)
    ax.set_facecolor(SURFACE); ax.grid(alpha=.55, color=GRID, lw=.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    lo, hi = zip(*[stats.beta.ppf([.025, .975], k+.5, n-k+.5) for k, n in zip(ks, ns)])
    err_lo = np.clip(np.array(ys)-np.array(lo), 0, None)
    err_hi = np.clip(np.array(hi)-np.array(ys), 0, None)
    ax.errorbar(xs, ys, yerr=[err_lo, err_hi],
                fmt='o-', color=C_CCE, lw=2.4, ms=10, capsize=5, zorder=5,
                label="CCE mul, by coverage")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8.5, color=INK2)
    if uni:
        lou, hiu = stats.beta.ppf([.025, .975], ku+.5, nu-ku+.5)
        ax.axhline(ku/nu, color=C_UNI, ls="--", lw=1.8)
        ax.axhspan(lou, hiu, color=C_UNI, alpha=.13)
        ax.text(max(xs)*.98, ku/nu+.02, f"uniform control  {ku/nu:.0%}  (n={nu})",
                ha="right", fontsize=9.5, color=INK2)
    ax.set_xscale("log"); ax.set_ylim(-.05, 1.05)
    ax.set_xlabel("replay coverage — % of buffer with a MEASURED score  (log)",
                  fontsize=11, color=INK2)
    ax.set_ylabel("fraction of seeds that SOLVED the map", fontsize=11, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    fig.text(.10, .955, "Does coverage CAUSE the win?", fontsize=19,
             fontweight="bold", color=INK, va="top")
    fig.text(.10, .885,
             "FrozenLake-det 8x8. Only n_score_sample varies; score_interval, rollouts, "
             "horizon and every training knob are held.\n"
             "Bars are 95% Jeffreys intervals. COMPLETED runs only — a crashed run would "
             "otherwise read as 'never solved'.",
             fontsize=10, color=INK2, va="top", linespacing=1.5)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
