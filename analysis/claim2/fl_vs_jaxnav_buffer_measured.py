"""Issue #18: FrozenLake's buffer coverage and zero-rate as the reference point.

Jeremy 09/03: "Compare frozen lake environment's buffer behavior and scoring share as a
reference point." The hypothesis on the call was that FL -- the one environment where CCE
clearly wins -- has a much larger scored share of the buffer.

Everything here is MEASURED from buffer snapshots, which carry consequence_scores and
scored_at, so measured entries can be separated from the inherited running mean. That
separation matters: reading score_n_zero out of ess.jsonl gives a zero-rate above 100% on
FrozenLake, because early in training FL's running mean is ITSELF 0 and the counter spans
the whole buffer, not just the scored part.

CAVEAT, and it belongs in anything this feeds: n=2 environments differing on many axes at
once cannot attribute the win/null difference to any one of them. This is a description,
not a cause.
"""
import glob
import os

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
OUT = os.path.join(WT, "docs/figures/real/claim2/jaxnav")
FL_C, JN_C = "#0d7c6b", "#a8341c"


def survey(root):
    dirs = sorted({os.path.dirname(os.path.dirname(f)) for f in
                   glob.glob(os.path.join(root, "*", "buffer", "buf_*.npz"))})
    cov, zero, medscore = [], [], []
    for i, d in enumerate(dirs):
        snaps = sorted(glob.glob(os.path.join(d, "buffer", "buf_*.npz")))
        try:
            z = np.load(snaps[-1])
        except Exception:
            continue
        cs, sa = np.asarray(z["consequence_scores"]), np.asarray(z["scored_at"])
        m = sa >= 0
        if m.sum() < 20:
            continue
        cov.append(100 * m.mean())
        zero.append(100 * float((cs[m] == 0).mean()))
        # PER-RUN median, not a pooled histogram: pooling the first N runs samples the
        # oldest directories and gave a median that contradicted the per-run figure.
        medscore.append(float(np.median(cs[m])))
    return np.array(cov), np.array(zero), np.array(medscore)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fl = survey(os.path.join(WT, "src/counterfactual_rl/agents/frozen_lake/runs"))
    jn = survey(os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), dpi=150)

    def strip(ax, a, b, title, note, xlabel):
        for vals, colour, lab, y in ((a, FL_C, "FrozenLake  (CCE WINS)", 1),
                                     (b, JN_C, "JaxNav  (CCE NULL)", 0)):
            jitter = y + np.random.default_rng(0).normal(0, .055, vals.size)
            ax.scatter(vals, jitter, s=16, alpha=.45, color=colour, lw=0)
            med = np.median(vals)
            ax.plot([med, med], [y - .19, y + .19], color=colour, lw=2.6)
            ax.annotate(f"median {med:.1f}%", (med, y + .26), ha="center",
                        fontsize=9.5, color=colour, weight="bold")
        ax.set_yticks([1, 0]); ax.set_yticklabels(["FrozenLake\nCCE WINS", "JaxNav\nCCE NULL"],
                                                  fontsize=9.5)
        ax.set_ylim(-.5, 1.55); ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11.5, loc="left", weight="bold", pad=8)
        ax.text(.5, -.28, note, transform=ax.transAxes, ha="center", fontsize=9.5,
                color="#55666f")
        ax.grid(alpha=.18, lw=.6, axis="x")

    strip(axes[0], fl[0], jn[0],
          "1.  Coverage — they OVERLAP",
          "the hypothesis on the 09/03 call. It does not hold.",
          "% of buffer holding a MEASURED score")
    axes[0].set_xscale("log")

    strip(axes[1], fl[1], jn[1],
          "2.  Dead scores — they SEPARATE",
          "this is where the two environments actually differ",
          "% of MEASURED scores that are exactly 0")

    # panel 3: the TYPICAL measured score in each run
    def strip3(ax, a, b):
        for vals, colour, y in ((a, FL_C, 1), (b, JN_C, 0)):
            jitter = y + np.random.default_rng(1).normal(0, .055, vals.size)
            ax.scatter(vals, jitter, s=16, alpha=.45, color=colour, lw=0)
            med = np.median(vals)
            ax.plot([med, med], [y - .19, y + .19], color=colour, lw=2.6)
            ax.annotate(f"median {med:.3f}", (med, y + .26), ha="center",
                        fontsize=9.5, color=colour, weight="bold")
        ax.set_yticks([1, 0]); ax.set_yticklabels(["FrozenLake\nCCE WINS",
                                                   "JaxNav\nCCE NULL"], fontsize=9.5)
        ax.set_ylim(-.5, 1.55); ax.set_xlim(-.05, 1.08)
        ax.set_xlabel("median measured score in that run", fontsize=10)
        ax.grid(alpha=.18, lw=.6, axis="x")
    strip3(axes[2], fl[2], jn[2])
    axes[2].set_title("3.  What a measured score SAYS", fontsize=11.5, loc="left",
                      weight="bold", pad=8)
    axes[2].text(.5, -.28, "FL saturates at the ceiling; JaxNav sits on the floor",
                 transform=axes[2].transAxes, ha="center", fontsize=9.5, color="#55666f")

    fig.suptitle("Issue #18 — FrozenLake vs JaxNav buffer, MEASURED from snapshots",
                 fontsize=13.5, x=.008, ha="left", weight="bold", y=.985)
    fig.text(.008, .915,
             f"FrozenLake n={fl[0].size} runs   ·   JaxNav n={jn[0].size} runs   ·   "
             "last snapshot of each   ·   measured entries only (scored_at >= 0)",
             fontsize=9.5, color="#55666f")
    fig.text(.008, .022,
             "Coverage was the hypothesis and it does NOT separate the two environments "
             "(FL median 5.2%, JaxNav 3.4%, ranges overlapping) — independently confirming the "
             "2026-09-09 retraction.\nWhat separates them is whether a score SAYS anything: "
             "FL's typical measured score is 1.000, JaxNav's is 0.000. "
             "CAVEAT: n=2 environments differ on many axes at once; this is a description, not a cause.",
             fontsize=8.8, color="#55666f")
    fig.tight_layout(rect=[0, .135, 1, .885])
    p = os.path.join(OUT, "fl_vs_jaxnav_buffer_measured.png")
    fig.savefig(p); print("wrote", p)
    for nm, a in (("FrozenLake", fl), ("JaxNav", jn)):
        print(f"  {nm:<12} n={a[0].size:<4} coverage med {np.median(a[0]):6.2f}%   "
              f"zeros med {np.median(a[1]):6.2f}%   per-run score med {np.median(a[2]):.3f}")


if __name__ == "__main__":
    main()
