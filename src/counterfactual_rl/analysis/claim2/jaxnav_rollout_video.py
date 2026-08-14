"""Render a trained JaxNav policy acting on random maps -- one GIF per arm.

Picks the best-performing seed in each arm of the 150k sweep, loads that run's
end-of-training weights (last.pkl), and rolls the greedy policy out on fresh
random maps drawn from the same distribution it trained and was evaluated on.

Every arm is rolled out on the SAME maps (the map keys are drawn once, up
front), so the three policies are compared like-for-like rather than each
getting its own lucky or unlucky draw.

Needs the run tree -- unlike the figures, this reads last.pkl weights, which
are far too large to commit, so it cannot fall back to the committed cache.
Job ids are in docs/figures/real/claim2/jaxnav/data/manifest_25seed_150k.json.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src JAX_PLATFORMS=cpu N_EPISODES=4 \
      python -m counterfactual_rl.analysis.claim2.jaxnav_rollout_video <outdir>
"""
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")

from counterfactual_rl.envs.jax_nav import JaxNavEnv
from counterfactual_rl.agents.jax_nav.dqn import _QNetwork
from counterfactual_rl.analysis.claim2 import jaxnav_holes_figures as F
from jaxmarl.environments.jaxnav import JaxNavVisualizer

RUNS = F.RUNS
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N_EPISODES = int(os.environ.get("N_EPISODES", "3"))
ROLLOUT_SEED = int(os.environ.get("ROLLOUT_SEED", "0"))


def best_seed(arm_jobs):
    """Job id with the highest final win rate in this arm."""
    scored = [(F.finals([j])[0], j) for j in arm_jobs if F.load(j) is not None]
    return max(scored)


def load_policy(job):
    with open(os.path.join(RUNS, str(job), "last.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    net = _QNetwork(
        obs_dim=205,
        hidden_dim=cfg.get("hidden_dim", 128),
        n_layers=cfg.get("n_layers", 3),
        n_actions=15,
    )
    params = jax.tree.map(jnp.array, ckpt["params"])
    return net, params, cfg


def build_env(cfg):
    return JaxNavEnv(
        map_id=cfg.get("map_id", "Grid-Rand-Poly"),
        map_size=tuple(cfg.get("map_size", (8, 8))),
        fill=cfg.get("fill", 0.1),
        goal_radius=cfg.get("goal_radius", 0.8),
        coll_rew=cfg.get("coll_rew", 0.0),
        max_steps=cfg.get("max_steps", 200),
        sparse_reward=cfg.get("sparse_reward", True),
    )


def rollout(env, net, params, key):
    """Greedy episode. Returns (obs dicts, states, rewards, reached_goal)."""
    raw = env._env
    obs_v, state = env.reset(key)
    # reward_seq is indexed by STATE frame in the visualizer, so it needs the
    # same length as state_seq -- pad the initial state with a 0 reward.
    obs_seq, state_seq, rew_seq = [raw.get_obs(state)], [state], [0.0]
    reached = False
    for _ in range(env.max_steps):
        act = int(jnp.argmax(net.apply(params, obs_v)))
        obs_v, state, r, done, _ = env.step(key, state, act)
        obs_seq.append(raw.get_obs(state))
        state_seq.append(state)
        rew_seq.append(float(r))
        reached = bool(np.asarray(state.goal_reached).ravel()[0])
        if bool(done):
            break
    return obs_seq, state_seq, rew_seq, reached


def frames_from_gif(path):
    from PIL import Image, ImageSequence
    im = Image.open(path)
    return [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]


def main():
    arms = F._power_arms(F.MANIFEST_150K)
    # Same map keys for every arm, so the three policies are compared on
    # identical maps rather than on whatever each happened to draw.
    map_keys = list(jax.random.split(jax.random.PRNGKey(ROLLOUT_SEED), N_EPISODES))
    labels = {"per": "PER", "cce_max": "CCE+max", "cce_wmean": "CCE+wmean"}
    os.makedirs(OUT, exist_ok=True)

    for arm in ("per", "cce_max", "cce_wmean"):
        score, job = best_seed(arms[arm])
        net, params, cfg = load_policy(job)
        env = build_env(cfg)
        print(f"\n{labels[arm]}: seed job {job}, final win rate {score*100:.1f}%")

        all_frames, outcomes = [], []
        for ep, k in enumerate(map_keys):
            obs_seq, state_seq, rew_seq, reached = rollout(env, net, params, k)
            outcomes.append(reached)
            steps = len(state_seq) - 1
            title = (f"{labels[arm]}  |  map {ep+1}  |  "
                     f"{'GOAL' if reached else 'FAILED'} in {steps} steps")
            viz = JaxNavVisualizer(
                env._env, obs_seq, state_seq, rew_seq,
                title_text=title, plot_lidar=False, plot_reward=False,
            )
            tmp = os.path.join(OUT, f"_tmp_{arm}_{ep}.gif")
            viz.animate(save_fname=tmp)
            fr = frames_from_gif(tmp)
            all_frames.extend(fr)
            all_frames.extend([fr[-1]] * 10)   # hold on the last frame
            os.remove(tmp)
            matplotlib.pyplot.close("all")
            print(f"   map {ep+1}: {steps:3d} steps, "
                  f"{'reached goal' if reached else 'NO goal'}")

        out = os.path.join(OUT, f"jaxnav_{arm}.gif")
        all_frames[0].save(out, save_all=True, append_images=all_frames[1:],
                           duration=60, loop=0, optimize=True)
        print(f"   wrote {out}  ({len(all_frames)} frames, "
              f"{sum(outcomes)}/{len(outcomes)} reached goal)")


if __name__ == "__main__":
    main()
