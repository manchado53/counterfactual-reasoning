"""What 'none' / 'minmax' / 'rank' priority normalisation actually do to the sampler.

Issue #16. Jeremy 08/20: "rerun with 0-1 normalization of TD errors -- equal exponents
do not imply equal contribution."

Read from REAL buffer snapshots, so this needs no GPU: the snapshot carries
consequence_scores, td_magnitudes and scored_at, which is everything the priority
computation consumes. Two snapshots are used on purpose -- the same question has a
different answer at 0.84% coverage and at 16%.

The plotted quantity is priority x N: 1.0 is uniform replay, 10 means "this slot is
drawn 10x more often than uniform". That is what the sampler actually does, as opposed
to the coefficient of variation of the raw magnitudes, which is scale-free and therefore
blind to min-max entirely.
"""
import glob
import json
import os
import subprocess

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
RUNS = os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs")
DATA = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data")
OUT = os.path.join(WT, "docs/figures/real/claim2/jaxnav")
EPS, BETA = 0.01, 0.25
MODES = [("none", "none  (shipped)", "#3f5763", "-"),
         ("minmax", "min-max", "#c98a5a", "--"),
         ("rank", "rank", "#0d7c6b", "-")]


def _probs(cs, td, mode):
    from counterfactual_rl.agents.shared.consequence_buffers import _normalise
    c, t = _normalise(cs.copy(), mode), _normalise(td.copy(), mode)
    pc = (c + EPS) ** BETA; pc /= pc.sum()
    pt = (t + EPS) ** BETA; pt /= pt.sum()
    comb = pc * pt                      # multiplicative, mu_c = mu_delta = 1
    return comb / comb.sum()


def _stats(cs, td):
    from counterfactual_rl.agents.shared.consequence_buffers import _normalise
    base = _probs(cs, td, "none")
    k = max(1, base.size // 100)
    top_base = set(np.argpartition(-base, k)[:k])
    rows = {}
    for mode, *_ in MODES:
        p = _probs(cs, td, mode)
        c, t = _normalise(cs.copy(), mode), _normalise(td.copy(), mode)
        rows[mode] = dict(
            p=p,
            ess=float(1.0 / np.sum(p ** 2) / p.size),
            overlap=len(set(np.argpartition(-p, k)[:k]) & top_base) / k,
            tv=float(0.5 * np.abs(p - base).sum()),
            cv_td=float(t.std() / t.mean()), cv_cce=float(c.std() / c.mean()),
            peak=float(p.max() * p.size))
    return rows


def _load(arm):
    m = json.load(open(os.path.join(DATA, "manifest_greedy_coverage.json")))
    arrays = sorted({v["array_id"] for v in m.values()})
    out = subprocess.run(["sacct", "-j", ",".join(arrays), "-X", "-n", "-P",
                          "--format=JobID,JobIDRaw"], capture_output=True, text=True).stdout
    raw = {l.split("|")[0]: l.split("|")[1] for l in out.strip().splitlines()}
    rid = raw.get(m[f"{arm}/0"]["job_id"])
    snaps = sorted(glob.glob(os.path.join(RUNS, rid, "buffer", "buf_*.npz")))
    if not snaps:
        return None
    z = np.load(snaps[-1])
    return (np.asarray(z["consequence_scores"], dtype=np.float64),
            np.asarray(z["td_magnitudes"], dtype=np.float64),
            np.asarray(z["scored_at"]), os.path.basename(snaps[-1]))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("greedy_base_max", "coverage 0.84%  (shipped rate)"),
              ("greedy_cov_max", "coverage 15.9%  (the new arm)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), dpi=150, sharey=True)
    for pi, (ax, (arm, subtitle)) in enumerate(zip(axes, panels)):
        got = _load(arm)
        if got is None:
            ax.text(.5, .5, f"{arm}\nno snapshot yet", ha="center", transform=ax.transAxes)
            continue
        cs, td, sa, fname = got
        rows = _stats(cs, td)
        n = cs.size
        for mode, lab, colour, ls in MODES:
            p = np.sort(rows[mode]["p"])[::-1] * n     # priority relative to uniform
            ax.plot(np.arange(1, n + 1), p, color=colour, ls=ls, lw=2.1,
                    label=f"{lab}   top slot gets {rows[mode]['peak']:.1f} tickets")
        ax.axhline(1.0, color="#a8341c", lw=1.2, ls=":")
        ax.text(1.3, 1.05, "1 ticket each = uniform replay (a fair lottery)",
                fontsize=8.5, color="#a8341c")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("slots, lined up LUCKIEST (left) -> UNLUCKIEST (right)")
        # Read-the-chart guides on the LEFT panel only. The lesson is identical in
        # both and the right panel has no free space that clears the stats box.
        pn = np.sort(rows["none"]["p"])[::-1] * n
        peak = pn[0]
        shelf_end = int(np.argmax(pn < 0.99 * peak)) or 1      # where the tie-mass ends
        if pi == 0:
          ax.annotate("higher = a few slots\nhog the replay",
                      xy=(1.2, peak * 0.97), xytext=(6, peak * 0.42),
                      fontsize=8.5, color="#55666f",
                      arrowprops=dict(arrowstyle="->", color="#55666f", lw=.9))
          ax.annotate("flat shelf = slots TIED at the top\n"
                      "(the batch just inserted, td = MAX)",
                      xy=(max(2, shelf_end // 3), peak * 1.005),
                      xytext=(max(6, shelf_end) * 2.2, peak * 0.58),
                      fontsize=8.5, color="#55666f",
                      arrowprops=dict(arrowstyle="->", color="#55666f", lw=.9))
        ax.set_title(subtitle, fontsize=10.5, loc="left", pad=6)
        ax.grid(alpha=.18, lw=.6, which="both")
        ax.legend(fontsize=8, loc="lower left")
        txt = (f"{fname}\n"
               f"top-1% slots kept vs 'none'\n"
               f"  min-max {100*rows['minmax']['overlap']:.0f}%"
               f"    rank {100*rows['rank']['overlap']:.0f}%\n"
               f"cv(TD) {rows['none']['cv_td']:.1f} -> rank {rows['rank']['cv_td']:.2f}\n"
               f"cv(CCE) {rows['none']['cv_cce']:.2f} -> rank {rows['rank']['cv_cce']:.2f}")
        ax.text(.97, .97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c5cfd6", alpha=.92))
    axes[0].set_ylabel("REPLAY TICKETS PER SLOT\n"
                       "(1.0 = drawn as often as everyone else)")
    fig.suptitle("How unfair is the replay lottery?   priority normalisation, issue #16",
                 fontsize=13, x=.01, ha="left", weight="bold")
    fig.text(.01, .935,
             "Each buffer slot is a lottery ticket holder. The line is tickets-per-slot, "
             "richest to poorest. Higher and steeper = more unfair.",
             fontsize=9.5, color="#55666f")
    fig.text(.01, .015,
             "Read: 'none' (shipped) gives the top slots 10x tickets. min-max squashes that to "
             "3x. rank is almost flat at 1.5x -- nearly a fair lottery.\n"
             "Note every mode has ESS 0.92-0.99, i.e. replay is close to uniform ALREADY, "
             "which caps how much this knob can matter.",
             fontsize=8.5, color="#55666f")
    fig.tight_layout(rect=[0, .055, 1, .915])
    out = os.path.join(OUT, "priority_normalisation.png")
    fig.savefig(out); print("wrote", out)
    for arm, _ in panels:
        got = _load(arm)
        if got is None: continue
        cs, td, sa, _f = got
        rows = _stats(cs, td)
        print(f"\n{arm}  coverage {(sa>=0).mean()*100:.2f}%")
        print(f"  {'mode':<9}{'ESS':>8}{'top-1% same':>13}{'TV':>9}{'peak/unif':>11}")
        for mode, *_ in MODES:
            r = rows[mode]
            print(f"  {mode:<9}{r['ess']:>8.4f}{100*r['overlap']:>12.1f}%{r['tv']:>9.4f}{r['peak']:>11.1f}")


if __name__ == "__main__":
    main()
