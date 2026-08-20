"""Side-by-side rollout videos for the ESS-matched balance sweep.

One GIF per map, all five balance arms playing the SAME map simultaneously, so
behaviour is compared frame-for-frame instead of across separate clips.

Seed choice is MEDIAN by final win rate, not best. The sweep's result is that
PER's weakness is unreliable seeds rather than a lower ceiling, so showing each
arm's best seed would hide exactly the thing worth looking at. Set
SEED_PICK=best to override.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src JAX_PLATFORMS=cpu \
      N_EPISODES=3 python -m counterfactual_rl.analysis.claim2.jaxnav_balance_video <outdir>
"""
import json
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
from PIL import Image, ImageSequence, ImageDraw

from counterfactual_rl.envs.jax_nav import JaxNavEnv
from counterfactual_rl.agents.jax_nav.dqn import _QNetwork
from counterfactual_rl.analysis.claim2 import jaxnav_holes_figures as F
from counterfactual_rl.analysis.claim2.jaxnav_balance_figures import _arms, _final
from jaxmarl.environments.jaxnav import JaxNavVisualizer

RUNS = F.RUNS
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N_EPISODES = int(os.environ.get("N_EPISODES", "3"))
ROLLOUT_SEED = int(os.environ.get("ROLLOUT_SEED", "0"))
SEED_PICK = os.environ.get("SEED_PICK", "median")


def pick_seed(jobs):
    scored = sorted((v, j) for j in jobs for v in [_final(j)] if v is not None)
    if not scored:
        return None, None
    return scored[-1] if SEED_PICK == "best" else scored[len(scored) // 2]


def load_policy(job):
    with open(os.path.join(RUNS, str(job), "last.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    net = _QNetwork(obs_dim=205, hidden_dim=cfg.get("hidden_dim", 128),
                    n_layers=cfg.get("n_layers", 3), n_actions=15)
    return net, jax.tree.map(jnp.array, ckpt["params"]), cfg


def build_env(cfg):
    return JaxNavEnv(
        map_id=cfg.get("map_id", "Grid-Rand-Poly"),
        map_size=tuple(cfg.get("map_size", (8, 8))),
        fill=cfg.get("fill", 0.1), goal_radius=cfg.get("goal_radius", 0.8),
        coll_rew=cfg.get("coll_rew", 0.0), max_steps=cfg.get("max_steps", 200),
        sparse_reward=cfg.get("sparse_reward", True))


def rollout(env, net, params, key):
    raw = env._env
    obs_v, state = env.reset(key)
    obs_seq, state_seq, rew_seq = [raw.get_obs(state)], [state], [0.0]
    reached = False
    for _ in range(env.max_steps):
        act = int(jnp.argmax(net.apply(params, obs_v)))
        obs_v, state, r, done, _ = env.step(key, state, act)
        obs_seq.append(raw.get_obs(state)); state_seq.append(state); rew_seq.append(float(r))
        reached = bool(np.asarray(state.goal_reached).ravel()[0])
        if bool(done):
            break
    return obs_seq, state_seq, rew_seq, reached


def render(env, obs_seq, state_seq, rew_seq, title, tmp):
    viz = JaxNavVisualizer(env._env, obs_seq, state_seq, rew_seq,
                           title_text=title, plot_lidar=False, plot_reward=False)
    viz.animate(save_fname=tmp)
    im = Image.open(tmp)
    fr = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
    im.close(); os.remove(tmp); matplotlib.pyplot.close("all")
    return fr


def main():
    arms = _arms()
    os.makedirs(OUT, exist_ok=True)
    map_keys = list(jax.random.split(jax.random.PRNGKey(ROLLOUT_SEED), N_EPISODES))

    policies = {}
    for bal, jobs in arms.items():
        score, job = pick_seed(jobs)
        if job is None:
            continue
        net, params, cfg = load_policy(job)
        policies[bal] = (net, params, build_env(cfg), job, score)
        print(f"balance {bal:<5} -> job {job} ({SEED_PICK} seed, final {score*100:.1f}%)")

    for ep, k in enumerate(map_keys):
        panels, summary = [], []
        for bal, (net, params, env, job, score) in policies.items():
            o, s, r, reached = rollout(env, net, params, k)
            steps = len(s) - 1
            lab = "0% CCE = PER" if bal == 0.0 else f"{int(bal*100)}% CCE"
            title = f"{lab}   {'GOAL' if reached else 'FAILED'} in {steps} steps"
            panels.append(render(env, o, s, r, title, os.path.join(OUT, f"_t{bal}_{ep}.gif")))
            summary.append(f"{lab}: {'goal' if reached else 'FAIL'} ({steps} st)")
            print(f"   map {ep+1}  {lab:<14} {'goal' if reached else 'NO goal':<8} {steps:3d} steps")

        n = max(len(p) for p in panels)
        panels = [p + [p[-1]] * (n - len(p)) for p in panels]   # hold finished arms
        w, h = panels[0][0].size
        frames = []
        for i in range(n):
            canvas = Image.new("RGB", (w * len(panels), h), "white")
            for j, p in enumerate(panels):
                canvas.paste(p[i], (j * w, 0))
            frames.append(canvas)
        frames.extend([frames[-1]] * 15)
        out = os.path.join(OUT, f"jaxnav_balance_map{ep+1}.gif")
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=60, loop=0, optimize=True)
        print(f"   wrote {out}  ({len(frames)} frames)  |  " + "  ".join(summary))


if __name__ == "__main__":
    main()
