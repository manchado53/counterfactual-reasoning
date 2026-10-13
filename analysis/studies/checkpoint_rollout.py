"""
Plan step 1 — how concentrated is the CCE replay priority, across the slip axis?

Theorem 3 is a statement about which transitions the buffer draws. Before
measuring covariances it is worth asking whether the deployed sampler is
concentrated at all: `p = (score + eps)^beta` with eps=0.01, beta=0.25 caps the
spread between any two transitions at ((1+eps)/eps)^beta = 3.17x, no matter what
the CCE scores are. If the realised sampler sits near uniform, the graded-slip
sweep tested near-uniform replay rather than the theory.

Reports, per (slip, seed, aggregation): the CCE score distribution, the shaped
priority spread, effective sample size, and Gini.

Scope and caveats
-----------------
- CCE component only. The deployed combined priority also carries a TD term;
  under multiplicative mixing the joint ceiling is 3.17^2 = 10.05x. Measuring
  that needs the 636-transition enumeration (step 2).
- ESS is over the distinct non-terminal states, not the live buffer's
  visitation-weighted composition. The 3.17x ceiling holds regardless.
- Last checkpoint per run only.
- Both aggregations are reported. `weighted_mean` under a greedy policy puts
  zero weight on every alternative, so `compute_consequence_metric` falls
  through to a plain mean -- the `mean` rows are what `weighted_mean` yields.

Usage
-----
    python -m counterfactual_rl.analysis.theorem3.priority_flatness           # measure + figure
    python -m counterfactual_rl.analysis.theorem3.priority_flatness --figure  # figure from cache
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN
from counterfactual_rl.analysis.metrics import compute_total_variation

EPS, BETA = 0.01, 0.25
CEILING = ((1.0 + EPS) ** BETA) / (EPS ** BETA)

SLIPS = ["0.0", "0.02", "0.04", "0.06", "0.08", "0.1", "0.133", "0.166",
         "0.333", "0.5", "0.666"]
ALGO = "consequence-dqn"
SEEDS_PER_SLIP = 3

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
OUT_DIR = os.path.join(_REPO, "docs", "figures", "theorem3_step1")


def _main_repo(path):
    """Worktrees live at <main>/.claude/worktrees/<name>; sibling worktrees are
    addressed from the main repo, not from inside another worktree."""
    marker = os.sep + os.path.join(".claude", "worktrees") + os.sep
    return path.split(marker)[0] if marker in path + os.sep else path


RUNS_DIR = os.environ.get(
    "GRADED_SLIP_RUNS",
    os.path.join(_main_repo(_REPO), ".claude", "worktrees", "graded-slip-frozenlake",
                 "src", "counterfactual_rl", "agents", "frozen_lake", "runs"),
)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


# --------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------
def read_header(metrics_log):
    """Parse the `# key: value` preamble a run writes at the top of metrics.log."""
    head = {}
    with open(metrics_log) as f:
        for line in f.read(2500).splitlines():
            if line.startswith("# ") and ":" in line:
                k, v = line[2:].split(":", 1)
                head[k.strip()] = v.strip()
    return head


def find_runs(slip, algo=ALGO, limit=SEEDS_PER_SLIP):
    """Last checkpoint of up to `limit` runs matching (slip, algo)."""
    found = []
    for d in sorted(glob.glob(os.path.join(RUNS_DIR, "*", ""))):
        log = os.path.join(d, "metrics.log")
        if not os.path.exists(log):
            continue
        h = read_header(log)
        if h.get("slip_prob") == slip and h.get("algorithm") == algo:
            cks = sorted(glob.glob(os.path.join(d, "checkpoints", "*.pkl")))
            if cks:
                found.append((h.get("seed"), cks[-1]))
        if len(found) >= limit:
            break
    return found


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def build_rollout_fn(env, network, horizon, gamma):
    """Triple-vmapped greedy rollout — mirrors consequence_dqn._build_rollout_fn."""
    def single_rollout(params, state_idx, first_action, rng_key):
        rng_key, step_key = jax.random.split(rng_key)
        _, next_state, reward, done, _ = env.step(step_key, state_idx, first_action)
        carry = (next_state, rng_key, reward, jnp.float32(gamma), done)

        def scan_step(c, _):
            s, key, cum, disc, done_flag = c
            action = jnp.argmax(network.apply(params, s))
            key, sk = jax.random.split(key)
            _, ns, r, nd, _ = env.step(sk, s, action)
            masked_r = jnp.where(done_flag, 0.0, r)
            return (ns, key, cum + disc * masked_r,
                    jnp.where(done_flag, disc, disc * gamma),
                    jnp.logical_or(done_flag, nd)), None

        final, _ = jax.lax.scan(scan_step, carry, xs=None, length=horizon - 1)
        return final[2]

    f = jax.vmap(single_rollout, in_axes=(None, None, None, 0))   # rollouts
    f = jax.vmap(f, in_axes=(None, None, 0, 0))                   # actions
    return jax.jit(jax.vmap(f, in_axes=(None, 0, None, 0)))       # states


def cce_scores(ckpt_path, n_rollouts=20, seed=0):
    """CCE score per non-terminal state, under both aggregations."""
    agent = FrozenLakeDQN.from_checkpoint(ckpt_path)   # env from ckpt; asserts slip
    env, cfg = agent.env, agent.config
    horizon = int(cfg.get("cf_horizon", 200))
    gamma = float(cfg.get("cf_gamma", 0.99))

    non_terminal = [r * env.ncols + c
                    for r, row in enumerate(env.desc)
                    for c, ch in enumerate(row) if ch not in ("H", "G")]

    rollout = build_rollout_fn(env, agent.network, horizon, gamma)
    B = len(non_terminal)
    keys = jax.random.split(jax.random.PRNGKey(seed),
                            B * 4 * n_rollouts).reshape(B, 4, n_rollouts, 2)
    returns = np.array(rollout(agent.params,
                               jnp.array(non_terminal, jnp.int32),
                               jnp.arange(4, dtype=jnp.int32), keys))

    out = {"max": [], "mean": []}
    for i, s in enumerate(non_terminal):
        greedy = int(jnp.argmax(agent.network.apply(agent.params, jnp.int32(s))))
        divs = [compute_total_variation(returns[i, greedy], returns[i, a])
                for a in range(4) if a != greedy]
        out["max"].append(max(divs))
        out["mean"].append(float(np.mean(divs)))
    return env.slip_prob, {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def shaped(scores):
    """Sampling distribution the buffer would use for these scores."""
    p = (np.asarray(scores) + EPS) ** BETA
    return p / p.sum()


def gini(q):
    x = np.sort(np.asarray(q, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def summarise(scores):
    c = np.asarray(scores)
    q = shaped(c)
    p = (c + EPS) ** BETA
    ess = 1.0 / np.sum(q ** 2)
    return dict(n=len(c), c_mean=float(c.mean()), c_std=float(c.std()),
                c_unique=int(len(np.unique(c))),
                c_frac_at_1=float(np.mean(c == 1.0)),
                p_ratio=float(p.max() / p.min()),
                ess=float(ess), ess_pct=float(100 * ess / len(c)),
                gini=gini(q))


def measure():
    print(f"analytic ceiling on priority spread: {CEILING:.3f}x "
          f"(eps={EPS}, beta={BETA}, score in [0,1])")
    print(f"multiplicative c*td ceiling: {CEILING ** 2:.2f}x\n")
    print(f"{'slip':>6} {'seed':>5} {'uniq':>5} {'@1.0':>6} {'c_mean':>7} "
          f"{'p_max/min':>10} {'ESS':>7} {'ESS%':>6} {'gini':>6}   agg")
    rows = []
    for slip in SLIPS:
        for seed, ckpt in find_runs(slip):
            try:
                env_slip, scores = cce_scores(ckpt)
            except Exception as e:                       # noqa: BLE001
                print(f"{slip:>6} {seed:>5}  SKIP {type(e).__name__}: {e}")
                continue
            for agg in ("max", "mean"):
                st = summarise(scores[agg])
                rows.append(dict(slip=env_slip, seed=seed, agg=agg, **st))
                print(f"{env_slip:>6.3f} {seed:>5} {st['c_unique']:>5} "
                      f"{st['c_frac_at_1']:>6.2f} {st['c_mean']:>7.3f} "
                      f"{st['p_ratio']:>10.2f} {st['ess']:>7.1f} "
                      f"{st['ess_pct']:>5.0f}% {st['gini']:>6.3f}   {agg}")
            sys.stdout.flush()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "step1_ess.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote step1_ess.json ({len(rows)} rows)")
    return rows


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def _hero_scores():
    """Per-state scores at the two extremes, cached (the rollouts are slow)."""
    cache = os.path.join(OUT_DIR, "hero_scores.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["det"], z["full"]
    det = cce_scores(find_runs("0.0", limit=1)[0][1])[1]["max"]
    full = cce_scores(find_runs("0.666", limit=1)[0][1])[1]["max"]
    np.savez(cache, det=det, full=full)
    return det, full


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(os.path.join(OUT_DIR, "step1_ess.json")))
    det, full = _hero_scores()

    def rel_uniform(c):
        q = shaped(c)
        return np.sort(q * len(q))[::-1]          # 1.0 == exactly uniform

    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.24,
                          left=0.075, right=0.975, top=0.86, bottom=0.09)

    ax = fig.add_subplot(gs[0, :])
    ax.axhspan(1.0, CEILING, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.text(1.5, CEILING - 0.07,
            f"headroom the priority never uses  —  ceiling {CEILING:.2f}×",
            ha="left", va="top", fontsize=10, color=INK2, style="italic")
    for arr, col, lab in ((det, BLUE, "deterministic  (slip 0)"),
                          (full, ORANGE, "full slip  (slip 0.666)")):
        ax.plot(np.arange(1, len(arr) + 1), rel_uniform(arr), lw=2, color=col,
                label=lab, zorder=3, solid_capstyle="round")
    ax.axhline(1.0, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.text(1.5, 0.95, "uniform replay", fontsize=9.5, color=INK2, va="top")
    ax.set_xlim(1, 53)
    ax.set_ylim(0, CEILING + 0.12)
    ax.set_xlabel("non-terminal states, ranked by priority", color=INK2, fontsize=10)
    ax.set_ylabel("sampling probability\n(× uniform)", color=INK2, fontsize=10)
    ax.set_title("What the replay sampler actually does", color=INK, fontsize=13,
                 fontweight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=10, loc="upper right", labelcolor=INK2)

    slips = sorted({r["slip"] for r in rows})
    xi = {s: i for i, s in enumerate(slips)}

    def panel(axp, key, title, ylab, ylim=None, hline=None, hlabel=None):
        for agg, col, off in (("max", BLUE, -0.11), ("mean", ORANGE, 0.11)):
            xs = [xi[r["slip"]] + off for r in rows if r["agg"] == agg]
            ys = [r[key] for r in rows if r["agg"] == agg]
            axp.scatter(xs, ys, s=42, color=col, alpha=0.85, lw=1.2,
                        edgecolor="white", zorder=3, label=f"{agg} aggregation")
        if hline is not None:
            axp.axhline(hline, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
            axp.text(len(slips) - 0.55, hline, hlabel, fontsize=9.5, color=INK2,
                     va="bottom", ha="right")
        axp.set_xticks(range(len(slips)))
        axp.set_xticklabels([f"{s:g}" for s in slips], fontsize=9, rotation=45)
        axp.set_xlabel("slip probability", color=INK2, fontsize=10)
        axp.set_ylabel(ylab, color=INK2, fontsize=10)
        axp.set_title(title, color=INK, fontsize=12, fontweight="bold",
                      loc="left", pad=8)
        if ylim:
            axp.set_ylim(*ylim)
        axp.set_xlim(-0.6, len(slips) - 0.4)

    panel(fig.add_subplot(gs[1, 0]), "ess_pct",
          "Every run, every slip level: still near-uniform",
          "effective sample size\n(% of uniform)", ylim=(0, 108),
          hline=100, hlabel="uniform = 100%")
    panel(fig.add_subplot(gs[1, 1]), "c_unique",
          "But the signal is blunt exactly where CCE won",
          "distinct CCE score values\n(out of 53 states)", ylim=(0, 57))

    for a in fig.axes:
        a.grid(axis="y", color=GRID, lw=0.8, alpha=0.7, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9)

    fig.axes[1].legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=INK2)
    fig.suptitle("CCE replay priority is nearly uniform at every noise level",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.075, ha="left", y=0.965)
    fig.text(0.075, 0.915,
             "FrozenLake 8×8 · consequence-dqn · last checkpoint · 3 seeds per slip · "
             "priority = (CCE + 0.01)$^{0.25}$",
             fontsize=10, color=INK2, ha="left")

    path = os.path.join(OUT_DIR, "fig_priority_flatness.png")
    fig.savefig(path, dpi=170, facecolor="white")
    print("wrote", path)


def figure_beta():
    """The exponent's effect on dynamic range — the one-slide version."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, full = _hero_scores()          # slip 0.666: 52 distinct CCE values

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.4, 5.1))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.76, bottom=0.13, wspace=0.26)

    # -- A: ceiling vs beta -------------------------------------------------
    betas = np.linspace(0.02, 1.0, 400)
    ratio = ((1.0 + EPS) / EPS) ** betas
    axa.plot(betas, ratio, lw=2.2, color=BLUE, zorder=3, solid_capstyle="round")
    axa.set_yscale("log")
    axa.axvline(BETA, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    axa.scatter([BETA], [CEILING], s=90, color=ORANGE, zorder=4,
                edgecolor="white", lw=1.6)
    axa.annotate(f"deployed\nβ = {BETA}  →  {CEILING:.2f}×",
                 xy=(BETA, CEILING), xytext=(BETA + 0.08, CEILING * 0.28),
                 fontsize=11, color=INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
    axa.annotate(f"raw score\nβ = 1.0  →  {((1+EPS)/EPS):.0f}×",
                 xy=(1.0, (1 + EPS) / EPS), xytext=(0.28, 52),
                 fontsize=10.5, color=INK2, va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2,
                                 shrinkA=6, shrinkB=6))
    axa.set_xlim(0, 1.04)
    axa.set_xlabel("β  (priority exponent)", color=INK2, fontsize=11)
    axa.set_ylabel("largest possible replay advantage\nbetween any two transitions",
                   color=INK2, fontsize=11)
    axa.set_title("The exponent sets a hard ceiling", color=INK, fontsize=12.5,
                  fontweight="bold", loc="left", pad=8)

    # -- B: the same real scores under two exponents ------------------------
    n = len(full)
    for beta, col, lab in ((BETA, BLUE, f"β = {BETA}   (deployed)"),
                           (1.0, ORANGE, "β = 1.0   (raw score)")):
        p = (full + EPS) ** beta
        q = p / p.sum()
        axb.plot(np.arange(1, n + 1), np.sort(q * n)[::-1], lw=2.2, color=col,
                 label=lab, zorder=3, solid_capstyle="round")
    axb.axhline(1.0, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    axb.text(n - 0.5, 1.06, "uniform replay", fontsize=9.5, color=INK2,
             ha="right", va="bottom")
    axb.set_yscale("log")
    axb.set_xlim(1, n)
    axb.set_xlabel("non-terminal states, ranked by priority", color=INK2, fontsize=11)
    axb.set_ylabel("sampling probability\n(× uniform)", color=INK2, fontsize=11)
    axb.set_title("Same CCE scores, two exponents", color=INK, fontsize=12.5,
                  fontweight="bold", loc="left", pad=8)
    axb.legend(frameon=False, fontsize=10.5, loc="lower left", labelcolor=INK2)

    for a in (axa, axb):
        a.grid(axis="y", color=GRID, lw=0.8, alpha=0.7, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9.5)

    fig.suptitle("β = 0.25 caps CCE's replay advantage at 3.2×, before μ ever sees it",
                 fontsize=15, fontweight="bold", color=INK, x=0.075, ha="left", y=0.945)
    fig.text(0.075, 0.855,
             "priority = (CCE score + 0.01)$^{β}$   ·   CCE ∈ [0,1]   ·   "
             "right panel: real scores, FrozenLake 8×8 at slip 0.666 (52 distinct values)",
             fontsize=10, color=INK2, ha="left")

    path = os.path.join(OUT_DIR, "fig_beta_ceiling.png")
    fig.savefig(path, dpi=170, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", action="store_true",
                    help="skip measurement, rebuild the figures from step1_ess.json")
    ap.add_argument("--beta-only", action="store_true",
                    help="rebuild only the beta-ceiling figure")
    args = ap.parse_args()
    if args.beta_only:
        figure_beta()
    else:
        if not args.figure:
            measure()
        figure()
        figure_beta()
