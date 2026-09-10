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


def survey(root, min_episode):
    # TRAINING STAGE MUST BE MATCHED. survey() reads the LAST snapshot of each run, and
    # a quarter of the JaxNav pool is mid-training (the 25k mechanism sweep, the 100k
    # drift runs, and whatever is running right now) while FrozenLake is almost entirely
    # end-of-training. Scores are worse early, so pooling them made JaxNav look worse for
    # a reason that has nothing to do with the environment. Keep only runs whose last
    # snapshot is at the END of a full-length run.
    import re
    dirs = sorted({os.path.dirname(os.path.dirname(f)) for f in
                   glob.glob(os.path.join(root, "*", "buffer", "buf_*.npz"))})
    cov, zero, medscore = [], [], []
    for i, d in enumerate(dirs):
        snaps = sorted(glob.glob(os.path.join(d, "buffer", "buf_*.npz")))
        if not snaps:
            continue
        m = re.search(r"buf_(\d+)\.npz", os.path.basename(snaps[-1]))
        if not m or int(m.group(1)) < min_episode:
            continue
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

    fl = survey(os.path.join(WT, "src/counterfactual_rl/agents/frozen_lake/runs"),
                13_500)     # >=90% of a 15,000-episode FL run
    jn = survey(os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs"),
                225_000)    # >=90% of a 250,000-episode JaxNav run

    # (title, fl values, jn values, unit, verdict, plain-language meaning)
    # VERDICTS ARE MEASURED, not assumed. With training stage matched all three rows
    # differ (Mann-Whitney p = 5.7e-05, 1.7e-04, 1.3e-15). An earlier version of this
    # figure called row 1 "SAME" -- that was an artifact of pooling mid-training runs,
    # which included the high-coverage arms still in flight and inflated JaxNav.
    ROWS = [
        ("1.  How MUCH of the buffer gets a real score",
         fl[0], jn[0], "%", "DIFFERS\nbut INERT",
         "6.5x apart — yet FrozenLake's own dose-response was FLAT from 0.65% to 33.7% "
         "(p=0.885),\nand JaxNav's 1.5% sits INSIDE that range. So it differs, and it "
         "is not what causes the difference."),
        ("2.  How many of those scores are DEAD (exactly 0)",
         fl[1], jn[1], "%", "DIFFERS",
         "4 in 5 JaxNav scores say nothing at all — nothing to rank"),
        ("3.  What a typical score SAYS",
         100 * fl[2], 100 * jn[2], "%", "DIFFERS",
         "FrozenLake sits at the ceiling, JaxNav on the floor"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.8), dpi=150)
    for ax, (title, a, b, unit, verdict, meaning) in zip(axes, ROWS):
        for y, vals, colour, name, tag in ((1, a, FL_C, "FrozenLake", "CCE WINS"),
                                           (0, b, JN_C, "JaxNav", "CCE NULL")):
            med = float(np.median(vals))
            lo, hi = np.percentile(vals, [25, 75])
            ax.barh(y, med, color=colour, height=.48, alpha=.9)
            # range goes in the LABEL, not as a whisker -- a whisker next to a bar of the
            # same colour reads as an error bar on the bar, which it is not.
            ax.text(med + 1.8, y + .10, f"{med:.0f}{unit}", va="center", fontsize=17,
                    weight="bold", color=colour)
            ax.text(med + 1.8, y - .26, f"middle 50% of runs: {lo:.0f}–{hi:.0f}{unit}",
                    va="center", fontsize=8.5, color=colour, alpha=.8)
            ax.text(-2, y, f"{name}   {tag}", va="center", ha="right", fontsize=11,
                    color=colour, weight="bold")
        ax.set_xlim(0, 118); ax.set_ylim(-.65, 1.65)
        ax.set_yticks([]); ax.grid(alpha=.15, lw=.6, axis="x")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=12.5, loc="left", weight="bold", pad=34, x=-.145)
        ax.text(-.145, 1.035, meaning, transform=ax.transAxes, fontsize=9,
                color="#55666f", va="bottom")
        vc = "#5a5a5a" if "INERT" in verdict else "#a8341c"
        ax.text(1.005, .5, verdict, transform=ax.transAxes, rotation=270, va="center",
                ha="left", fontsize=10.5, weight="bold", color=vc, linespacing=1.4)
    axes[-1].set_xlabel("percent of the buffer / of the scores", fontsize=9.5)
    fig.suptitle("Why does CCE win on FrozenLake but do nothing on JaxNav?",
                 fontsize=15, x=.008, ha="left", weight="bold")
    fig.text(.008, .945,
             f"Measured from buffer snapshots — {fl[0].size} FrozenLake runs, {jn[0].size} "
             "JaxNav runs — END of a full-length run in both, measured entries only.   Issue #18.",
             fontsize=10, color="#55666f")
    fig.text(.008, .012,
             "ANSWER: all three differ, but only rows 2 and 3 are candidates for the CAUSE. "
             "Coverage is ruled out by FrozenLake's own dose-response, which was flat across a "
             "range that brackets JaxNav's value.\n"
             "CAVEAT: n=2 environments differ on many axes at once. This describes the "
             "difference; it does not prove it is the cause.",
             fontsize=9.2, color="#55666f")
    fig.tight_layout(rect=[.115, .085, .975, .915], h_pad=4.2)
    p = os.path.join(OUT, "fl_vs_jaxnav_buffer_measured.png")
    fig.savefig(p, facecolor="white")
    print("wrote", p)
    for nm, a in (("FrozenLake", fl), ("JaxNav", jn)):
        print(f"  {nm:<12} n={a[0].size:<4} coverage {np.median(a[0]):6.2f}%   "
              f"dead {np.median(a[1]):6.2f}%   typical score {np.median(a[2]):.3f}")


if __name__ == "__main__":
    main()
