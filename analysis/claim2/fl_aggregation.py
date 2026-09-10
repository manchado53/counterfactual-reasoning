"""Does the FrozenLake headline survive a GRADED aggregation?  (issue #3)

FrozenLake calls compute_consequence_metric WITHOUT action_probs
(frozen_lake/consequence_dqn.py:167), so its configured 'weighted_mean' falls through to
the `else: return max(divergences.values())` branch. Verified by direct execution: FL's
exact call returns 0.850000, identical to aggregation='max', vs 0.825000 for a real
weighted mean. The 208-run headline is therefore A WIN UNDER MAX, labelled otherwise.

'mean' is the cheap test: it needs no action_probs and so no code change, and it asks the
load-bearing question -- does the win need the MAXIMUM divergence across alternative
actions, or just a graded score?

SOLVED = reached >= 95% win rate at ANY eval. That is the criterion the 82% headline uses,
and it is why the preset sets early_stop_win_rate=None. A trailing mean is meaningless
here: FL-det's win rate oscillates 0% / 100% between evals.
"""
import glob
import json
import os

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
FL = os.path.join(WT, "src/counterfactual_rl/agents/frozen_lake/runs")
EXP = os.path.join(WT, "src/counterfactual_rl/agents/frozen_lake/experiments")
OUT = os.path.join(WT, "docs/figures/real/claim2/jaxnav")
TH = 95.0
C_MAX, C_MEAN, C_UNI = "#0d7c6b", "#c0392b", "#5a5a5a"


def curve(jid):
    p = os.path.join(FL, str(jid), "metrics.log")
    if not os.path.exists(p):
        return None
    w = []
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if len(t) >= 4:
            try:
                w.append(float(t[3].rstrip("%")))
            except ValueError:
                pass
    return np.array(w) if len(w) >= 10 else None


def solved(jid):
    c = curve(jid)
    return None if c is None else bool(c.max() >= TH)


def gather():
    new, ctrl, uni = {}, {}, {}
    m = json.load(open(sorted(glob.glob(os.path.join(EXP, "*/fl_aggregation*/*.json")))[-1]))
    for j, c in m.items():
        s = solved(j)
        if s is not None:
            new[int(c["seed"])] = s
    for p in glob.glob(os.path.join(EXP, "*/*/*.json")):
        if "fl_aggregation" in p:
            continue
        try:
            mm = json.load(open(p))
        except Exception:
            continue
        if not isinstance(mm, dict):
            continue
        for j, c in mm.items():
            if not isinstance(c, dict) or c.get("is_slippery") is not False:
                continue
            if c.get("n_episodes") != 15_000:
                continue
            if (c.get("algorithm") == "consequence-dqn" and c.get("n_score_sample") == 128
                    and c.get("priority_mixing") == "multiplicative"
                    and c.get("consequence_aggregation") in (None, "weighted_mean")):
                s = solved(j)
                if s is not None:
                    ctrl.setdefault(int(c.get("seed", -1)), s)
            if c.get("algorithm") == "dqn-uniform":
                s = solved(j)
                if s is not None:
                    uni.setdefault(int(c.get("seed", -1)), s)
    return ctrl, new, uni


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import fisher_exact

    ctrl, new, uni = gather()
    rows = [("CCE  aggregation = MAX\n(what the paper actually ran)", ctrl, C_MAX),
            ("CCE  aggregation = 'mean'\n(the graded alternative)", new, C_MEAN),
            ("uniform  (no CCE)", uni, C_UNI)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=150,
                                 gridspec_kw=dict(width_ratios=[1, 1.5]))

    ys = np.arange(len(rows))[::-1]
    for y, (lab, d, colour) in zip(ys, rows):
        v = np.array(list(d.values()))
        if not v.size:
            continue
        rate = 100 * v.mean()
        a1.barh(y, rate, color=colour, alpha=.85, height=.55)
        a1.text(rate + 1.5, y, f"{v.sum()}/{v.size} = {rate:.0f}%", va="center",
                fontsize=11, color=colour, weight="bold")
    a1.set_yticks(ys); a1.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    a1.set_xlim(0, 118); a1.set_xlabel("% of seeds that SOLVED  (reached ≥95% win rate)",
                                       fontsize=10)
    a1.grid(alpha=.2, lw=.6, axis="x")
    a1.set_title("Solve rate", fontsize=12, loc="left", weight="bold")

    seeds = sorted(set(ctrl) | set(new))
    for y, (lab, d, colour) in zip(ys, rows[:2]):
        for i, s in enumerate(seeds):
            if s not in d:
                continue
            a2.add_patch(plt.Rectangle((i - .42, y - .32), .84, .64,
                                       facecolor=colour if d[s] else "#ffffff",
                                       edgecolor=colour, lw=1.3, alpha=.9 if d[s] else 1))
    disagree = [i for i, s in enumerate(seeds) if s in ctrl and s in new and ctrl[s] != new[s]]
    for i in disagree:
        a2.plot([i, i], [ys[1] - .55, ys[0] + .55], color="#a8341c", lw=1, ls=":", zorder=0)
    a2.set_xlim(-1, len(seeds)); a2.set_ylim(ys[1] - .8, ys[0] + .8)
    a2.set_yticks(ys[:2]); a2.set_yticklabels(["MAX", "'mean'"], fontsize=11)
    a2.set_xticks(range(0, len(seeds), 2)); a2.set_xticklabels([seeds[i] for i in range(0, len(seeds), 2)],
                                                              fontsize=8.5)
    a2.set_xlabel("seed", fontsize=10)
    a2.set_title(f"Per-seed outcome   ·   filled = solved   ·   "
                 f"{len(disagree)} of {len(seeds)} seeds disagree (dotted)",
                 fontsize=12, loc="left", weight="bold")
    for sp in ("top", "right", "left"):
        a2.spines[sp].set_visible(False)

    a = np.array(list(ctrl.values())); b = np.array(list(new.values()))
    p = fisher_exact([[b.sum(), b.size - b.sum()], [a.sum(), a.size - a.sum()]])[1]
    fig.suptitle("Does the FrozenLake win survive a graded aggregation?   issue #3",
                 fontsize=13.5, x=.008, ha="left", weight="bold")
    fig.text(.008, .015,
             f"FrozenLake deterministic 8×8, n_score_sample 128, 26 seeds each.  "
             f"SOLVED = reached ≥95% win rate at any eval (the criterion the headline uses; "
             f"a trailing mean is meaningless as FL-det oscillates 0%/100%).\n"
             f"Fisher exact, mean vs max: p = {p:.3f}.  Not significant at n=26 — but it is a "
             f"{100*(a.mean()-b.mean()):.0f}-point drop in the wrong direction on the result the "
             f"paper rests on.\n"
             f"FrozenLake passes no action_probs, so its configured 'weighted_mean' has always "
             f"executed as MAX.  The headline is a win under max, labelled otherwise.",
             fontsize=8.8, color="#55666f")
    fig.tight_layout(rect=[0, .105, 1, .93])
    q = os.path.join(OUT, "fl_aggregation.png")
    fig.savefig(q); print("wrote", q)
    for lab, d, _ in rows:
        v = np.array(list(d.values()))
        if v.size:
            print(f"  {lab.splitlines()[0]:<40} {v.sum():>3}/{v.size:<3} = {100*v.mean():5.1f}%")
    print(f"  Fisher p = {p:.4f}   seeds disagreeing: {len(disagree)}")


if __name__ == "__main__":
    main()
