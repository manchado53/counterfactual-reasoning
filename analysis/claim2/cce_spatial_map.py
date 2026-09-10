"""Spatial map of CCE score over a JaxNav world, at the end of training.

The project has a FrozenLake version of this (the Claim-1 heatmaps) but never a JaxNav
one, because JaxNav states are continuous and the buffer snapshots store only the
priority arrays -- not the states -- so it cannot be reconstructed after the fact.

This does it directly instead: load a trained checkpoint, freeze ONE map and ONE goal,
sweep the robot across a grid of positions, and score each position with exactly the
scoring path training uses (JaxNavConsequenceDQN._score_states, added for the drift
probe -- it takes arbitrary states and touches no buffer).

WHAT THE COLOURS MEAN. CCE asks "did the action choice matter here?" -- high where the
15 actions lead to visibly different returns, low where they all end the same way. So
the prediction is: high near obstacles and near the goal approach, low in open space
and deep inside dead ends where everything fails equally.

ONE MAP, ONE GOAL. JaxNav randomises the map every episode (~800 distinct maps live in
a full buffer), so there is no single "the" map -- this is one draw, shown to make the
structure legible, not to average over worlds.
"""
import argparse
import glob
import os

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
RUNS = os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs")
OUT = os.path.join(WT, "docs/figures/real/claim2/jaxnav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir id; default = a finished greedy_cov_max seed")
    ap.add_argument("--grid", type=int, default=34, help="positions per axis")
    ap.add_argument("--map-seed", type=int, default=3)
    args = ap.parse_args()

    import jax, jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from counterfactual_rl.agents.jax_nav.consequence_dqn_vectorized import (
        JaxNavConsequenceDQNVectorized)

    run = args.run
    if run is None:
        cands = [d for d in sorted(glob.glob(os.path.join(RUNS, "*")))
                 if os.path.exists(os.path.join(d, "best.pkl"))]
        run = os.path.basename(cands[-1])
    ckpt = os.path.join(RUNS, run, "best.pkl")
    import pickle
    cfg = pickle.load(open(ckpt, "rb"))["config"]
    cfg = {**cfg, "n_envs": 8, "collect_steps": 4, "n_episodes": 1, "vectorized": True}
    agent = JaxNavConsequenceDQNVectorized(cfg)
    agent.load(ckpt)
    print(f"loaded {ckpt}\n  metric={agent.consequence_metric}  agg={agent.consequence_aggregation}"
          f"  T={agent.cf_rollout_temperature}  R={agent.cf_n_rollouts}  H={agent.cf_horizon}")

    env = agent.env
    _, base = env.reset(jax.random.PRNGKey(args.map_seed))
    lo, hi = 1.0, float(np.asarray(base.map_data).shape[0]) - 1.0
    xs = np.linspace(lo, hi, args.grid)
    ys = np.linspace(lo, hi, args.grid)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], 1)

    # one state per grid point: same map, same goal, robot moved
    def at(p):
        return base.replace(pos=jnp.asarray(p, dtype=jnp.float32).reshape(1, 2))
    states = [at(p) for p in pts]
    batched = jax.tree.map(lambda *l: jnp.stack(l), *states)
    obs = [np.asarray(env.get_obs(s)).reshape(-1) for s in states]
    q = jax.vmap(agent.network.apply, in_axes=(None, 0))(
        agent.params, jnp.asarray(np.stack(obs), dtype=jnp.float32))
    taken = [int(x) for x in np.argmax(np.asarray(q), axis=-1)]   # the greedy action

    scores, _ = agent._score_states(batched, obs, taken)
    S = np.asarray(scores).reshape(args.grid, args.grid)
    grid = np.asarray(base.map_data)
    goal = np.asarray(base.goal).reshape(-1)[:2]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=150)
    for ax, (dat, title, cmap) in zip(axes, [
            (S, "CCE score  —  'did the action choice matter here?'", "magma"),
            (np.max(np.asarray(q), axis=-1).reshape(args.grid, args.grid),
             "max Q  —  'how good is it here?'   (for contrast)", "viridis")]):
        im = ax.imshow(dat, origin="lower", cmap=cmap,
                       extent=[lo, hi, lo, hi], aspect="equal", interpolation="bilinear")
        oy, ox = np.where(grid > 0)
        ax.scatter(ox + .5, oy + .5, s=170, marker="s", c="white", edgecolors="black",
                   linewidths=.8, label="obstacle")
        ax.scatter(*goal, s=300, marker="*", c="#39ff88", edgecolors="black",
                   linewidths=1.1, zorder=5, label="goal")
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=.85)
    fig.suptitle(f"Where does CCE think the action matters?   JaxNav 8×8, end of training "
                 f"(run {run})", fontsize=13, x=.008, ha="left", weight="bold")
    fig.text(.008, .015,
             f"{args.grid}×{args.grid} robot positions on ONE fixed map and goal, each scored with "
             f"the SAME path training uses (metric={agent.consequence_metric}, "
             f"agg={agent.consequence_aggregation}, T={agent.cf_rollout_temperature}, "
             f"R={agent.cf_n_rollouts}, H={agent.cf_horizon}).\n"
             "JaxNav randomises the map every episode, so this is ONE draw shown to make the "
             "structure legible — not an average over worlds. Right panel is the value function "
             "for contrast: CCE is not supposed to look like it.",
             fontsize=8.8, color="#55666f")
    fig.tight_layout(rect=[0, .075, 1, .93])
    p = os.path.join(OUT, "cce_spatial_map.png")
    fig.savefig(p, facecolor="white")
    print("wrote", p)
    print(f"  CCE score: min {S.min():.4f}  median {np.median(S):.4f}  max {S.max():.4f}"
          f"  zeros {100*(S==0).mean():.1f}%")


if __name__ == "__main__":
    main()
