"""When does the CCE signal exist? Score spread over training + its distribution.

The score's relative spread (std/mean) collapses across training except for one
bump around episodes 35k-50k, where the policy first starts reaching the goal.
This was built to ask whether that bump is a brief BIMODAL window -- a few states
carrying real consequence while the rest are blank.

WHAT IT ACTUALLY SHOWS. Without bootstrap the score is mostly zero at EVERY
checkpoint (83.6 / 98.0 / 82.4 / 57.0 / 78.1 percent, 10k -> 250k), so there is
no "populated" phase to bump out of and back into. The bump coincides with the
zero share falling 98% -> 82% around the win-rate trough, which is consistent
with a few states gaining signal, but the shape claim is NOT established here --
see the power limit below.

Two panels:
  top     score spread from ess.jsonl, median of 5 seeds, log axis.
  bottom  share of states scoring exactly 0 (bar) and the distribution of the
          NONZERO scores (curve), at five checkpoints.

WHAT THIS IS NOT. The bottom panel scores states sampled ON-POLICY at each
checkpoint; it is not the distribution of the scores STORED in the replay
buffer. The buffer is never checkpointed, so that one cannot be recovered after
the fact -- and it would look different, because ~98.4% of buffer entries never
receive a measured score and carry the running mean they inherited on insertion
(consequence_buffers.py:102). Landing a histogram in `priority_diagnostics()` is
what makes the stored distribution available for future runs (issue #8).

POWER LIMIT. At 256 states a row can carry as few as 5 nonzero scores (OFF arm
at 25k). The zero-share bars are sound -- SE on a proportion is about +/-2.4pp --
but the nonzero curves are indicative only. Raise N_STATES to ~2048 before
reading anything into their shape.

    # step 1, needs a GPU
    python -m counterfactual_rl.analysis.claim2.jaxnav_signal_birth --compute
    # step 2, anywhere
    python -m counterfactual_rl.analysis.claim2.jaxnav_signal_birth
"""
import argparse
import glob
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .jaxnav_bootstrap_curves import resolve, MANIFEST, RUNS, ROOT

OUT = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/fig_signal_birth.png")
CACHE = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/signal_birth.json")

CELL = "8x8_f03"
ARMS = [("cce_wmean", "bootstrap OFF"), ("cce_wmean_bs", "bootstrap ON")]
TARGETS = [10_000, 25_000, 45_000, 100_000, 250_000]
N_STATES, CHUNK = 256, 128          # 128 x 15 actions x 20 rollouts = 38,400 lanes,
                                    # the largest batch measured safe on a T4
TARGET, DECAY = 250_000, 62_500
C_OFF, C_ON = "#0072b2", "#d55e00"
INK, INK2, GRID, SUNK = "#0b0b0b", "#52514e", "#dcdad5", "#f0eeea"
HOT = "#b0521f"


# ----------------------------------------------------------------- compute
def compute(array_id):
    import jax
    import jax.numpy as jnp
    import pickle
    from counterfactual_rl.agents.jax_nav.consequence_dqn import JaxNavConsequenceDQN
    from .jaxnav_score_probe import collect_states, per_state_scores

    man = json.load(open(MANIFEST))
    resolved = resolve(array_id)
    out = {}
    for arm, _ in ARMS:
        rec = man[f"{CELL}/{arm}/0"]
        run_dir = resolved[rec["job_id"]][0]
        ckdir = os.path.join(RUNS, run_dir, "checkpoints")
        avail = sorted(int(re.search(r"ckpt_(\d+)", f).group(1))
                       for f in glob.glob(os.path.join(ckdir, "ckpt_*.pkl")))
        out[arm] = {}
        for want in TARGETS:
            ep = min(avail, key=lambda c: abs(c - want))
            blob = pickle.load(open(os.path.join(ckdir, f"ckpt_{ep:07d}.pkl"), "rb"))
            cfg = dict(blob["config"])
            agent = JaxNavConsequenceDQN(cfg)
            agent.params = blob["params"]
            agent.target_params = blob.get("target_params", blob["params"])
            agent._build_rollout_fn()
            eps = float(blob.get("epsilon", 0.05))
            A, N = agent.n_actions, agent.cf_n_rollouts
            scores = []
            for c in range(N_STATES // CHUNK):     # chunked to stay under the T4 ceiling
                sb = collect_states(agent, CHUNK, jax.random.PRNGKey(100 + c), eps)
                keys = jax.random.split(jax.random.PRNGKey(200 + c),
                                        CHUNK * A * N).reshape(CHUNK, A, N, 2)
                R = np.array(agent._compiled_rollout_fn(
                    agent.params, agent.target_params, sb, agent._all_actions, keys))
                scores.append(per_state_scores(R))
            s = np.concatenate(scores)
            out[arm][str(ep)] = [round(float(v), 5) for v in s]
            print(f"  {arm:14s} ep {ep:>7,}  eps {eps:.3f}  n={len(s)}  "
                  f"median {np.median(s):.4f}  zero {np.mean(s == 0)*100:.1f}%",
                  flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"))
    print(f"\nwrote {CACHE}")


# -------------------------------------------------------------------- plot
def spread_trace(array_id, arm):
    """median std/mean across the arm's 5 seeds, from ess.jsonl."""
    man = json.load(open(MANIFEST))
    resolved = resolve(array_id)
    g = np.linspace(500, TARGET, 300)
    rows = []
    for s in range(5):
        rec = man.get(f"{CELL}/{arm}/{s}")
        if not rec or rec["job_id"] not in resolved:
            continue
        f = os.path.join(RUNS, resolved[rec["job_id"]][0], "ess.jsonl")
        if not os.path.exists(f):
            continue
        e, cv = [], []
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r["score_mean"]
            e.append(r["episode"]); cv.append(r["score_std"] / m if m else np.nan)
        if len(e) > 5:
            rows.append(np.interp(g, e, cv))
    return g, np.median(rows, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="274476")
    ap.add_argument("--compute", action="store_true", help="score checkpoints (needs a GPU)")
    a = ap.parse_args()
    if a.compute:
        compute(a.array)
        return

    data = json.load(open(CACHE))
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=.30,
                          left=.065, right=.985, top=.815, bottom=.065)

    # ---- top: the spread trace, with the three phases -------------------
    ax = fig.add_subplot(gs[0])
    for arm, lab in ARMS:
        g, cv = spread_trace(a.array, arm)
        ax.plot(g / 1000, cv, lw=2.2, color=C_OFF if arm == "cce_wmean" else C_ON,
                label=f"{lab}   n=5")
    ax.axvspan(0, DECAY / 1000, color=SUNK, zorder=0)
    ax.axvspan(35, 52, color=HOT, alpha=.12, zorder=0)
    ax.set_yscale("log"); ax.set_xlim(0, TARGET / 1000); ax.set_ylim(.06, 20)
    ax.set_ylabel("score spread   std / mean", fontsize=10.5, color=INK2)
    ax.set_xlabel("episodes (thousands)", fontsize=10, color=INK2)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    g, cv = spread_trace(a.array, "cce_wmean")
    w = (g > 30000) & (g < 60000)
    pk = g[w][np.argmax(cv[w])] / 1000
    ax.annotate(f"peak  {cv[w].max():.2f}", xy=(pk, cv[w].max()),
                xytext=(pk + 26, 6.0), fontsize=10, color=HOT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=HOT, lw=1.3))
    ax.text(31, .085, "epsilon decay", fontsize=9, ha="center", color=INK2,
            style="italic")
    ax.text(43.5, 11, "bump", fontsize=9.5, ha="center", color=HOT,
            fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.7, alpha=.7); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)

    # ---- bottom: zero mass split OUT from the nonzero shape --------------
    # A single histogram would be swamped: 82% of the OFF arm's mass sits in the
    # zero bin, flattening the tail into invisibility. So the zero fraction gets
    # its own bar and only the NONZERO scores are binned. n is printed per row
    # because with 256 states some rows have very few nonzero points and their
    # shape must not be read as signal.
    ax2 = fig.add_subplot(gs[1])
    # each arm's checkpoints land on slightly different episodes (10,187 vs
    # 10,210), so rows are matched by POSITION and labelled with the target.
    keys = {arm: sorted(data[arm], key=int) for arm, _ in ARMS}
    eps_list = TARGETS
    bins = np.linspace(0, 0.8, 26)
    ctr = (bins[:-1] + bins[1:]) / 2
    ZW = 0.26                       # width of the zero-bar zone, in x units
    X0 = -0.40                      # where the zero zone starts
    STEP, HEIGHT = 1.0, 0.80
    for row, ep in enumerate(eps_list):
        base = (len(eps_list) - 1 - row) * STEP
        ax2.axhline(base, color=GRID, lw=.8, zorder=1)
        for k, (arm, _) in enumerate(ARMS):
            s_all = np.array(data[arm][keys[arm][row]])
            col = C_OFF if arm == "cce_wmean" else C_ON
            zfrac = float(np.mean(s_all == 0))
            nz = s_all[s_all > 0]
            # zero bar
            ax2.barh(base + .12 + k * .30, ZW * zfrac, height=.24, left=X0,
                     color=col, alpha=.75, zorder=3)
            ax2.text(X0 + ZW * zfrac + .012, base + .12 + k * .30,
                     f"{zfrac*100:.0f}%", fontsize=8.5, va="center", color=col)
            # nonzero shape
            if len(nz) >= 3:
                h, _ = np.histogram(np.clip(nz, 0, bins[-1]), bins=bins)
                h = h / h.max()
                ax2.fill_between(ctr, base, base + h * HEIGHT, color=col,
                                 alpha=.26, lw=0, zorder=2)
                ax2.plot(ctr, base + h * HEIGHT, color=col, lw=1.8, zorder=3)
            ax2.text(0.815, base + .12 + k * .30, f"n={len(nz)}", fontsize=8.5,
                     va="center", color=col, family="DejaVu Sans")
        ax2.text(X0 - .03, base + .30, f"{ep/1000:.0f}k", fontsize=11.5, ha="right",
                 color=HOT if 35000 < ep < 52000 else INK, fontweight="bold")
    ax2.axvline(0, color=GRID, lw=1)
    ax2.set_xlim(X0 - .12, 0.90); ax2.set_ylim(-.14, len(eps_list) * STEP)
    ax2.set_yticks([])
    ax2.set_xticks([0, .2, .4, .6, .8])
    ax2.set_xlabel("per-state CCE score, NONZERO scores only   "
                   "(bar at left = share scoring exactly 0)", fontsize=10, color=INK2)
    ax2.text(X0, len(eps_list) * STEP - .12, "share at zero", fontsize=9,
             color=INK2)
    ax2.text(0.815, len(eps_list) * STEP - .12, "nonzero pts", fontsize=9, color=INK2)
    ax2.text(X0 - .03, len(eps_list) * STEP - .12, "episode", fontsize=9,
             ha="right", color=INK2)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(colors=INK2, labelsize=9)

    fig.text(.065, .955, "When does the CCE signal actually exist?",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(.065, .921,
             "JaxNav 8\u00d78 fill 0.3, mul\u00b7wmean, seed 0. Top: relative spread of the "
             "score across the buffer (ess.jsonl, median of 5 seeds, log axis).",
             fontsize=10, color=INK2)
    fig.text(.065, .896,
             "Bottom: per-state scores re-computed from checkpoints \u2014 256 on-policy "
             "states each. Blue = bootstrap OFF, orange = ON.",
             fontsize=10, color=INK2)
    fig.text(.065, .872,
             "TWO LIMITS. (1) These are scores computed FRESH from rollouts, not the "
             "values STORED in the buffer \u2014 the buffer is never checkpointed (issue #8).",
             fontsize=10, color=HOT)
    fig.text(.065, .849,
             "(2) With only 5\u2013244 nonzero points per row, the curve SHAPES are "
             "indicative, not evidence. Read the zero bars, which are solid to about "
             "\u00b12pp.",
             fontsize=10, color=HOT)

    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
