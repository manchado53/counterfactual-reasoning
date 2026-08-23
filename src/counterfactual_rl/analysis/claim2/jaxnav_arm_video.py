"""Side-by-side rollouts of each ARM's final policy on the same maps.

Built to answer: do the replay strategies produce visibly different behaviour,
or the same policy reached by different routes?

Uses the completed 500k sweep (jobs 272275-272374), where uniform / PER /
CCE+max / CCE+wmean all ran 25 seeds to the full budget. The 6-cell factorial's
CCE arms are only partly trained, so their policies are not final yet.

Every arm plays the SAME maps, drawn once up front, and all panels advance in
lockstep -- an arm that finishes early freezes on its last frame while the
others keep going, so a frozen panel means "already arrived".

MEDIAN seed per arm, not best. The recurring finding in this project is that
replay strategy shows up in the BAD seeds, not the ceiling, so best-seed
rendering would hide the thing worth looking at. Set SEED_PICK=best to override.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src JAX_PLATFORMS=cpu \
      N_MAPS=5 python -m counterfactual_rl.analysis.claim2.jaxnav_arm_video <outdir>
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
from PIL import Image, ImageSequence

from counterfactual_rl.envs.jax_nav import JaxNavEnv
from counterfactual_rl.agents.jax_nav.dqn import _QNetwork
from counterfactual_rl.analysis.claim2 import jaxnav_holes_figures as F
from jaxmarl.environments.jaxnav import JaxNavVisualizer

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N_MAPS = int(os.environ.get("N_MAPS", "5"))
ROLLOUT_SEED = int(os.environ.get("ROLLOUT_SEED", "7"))
SEED_PICK = os.environ.get("SEED_PICK", "median")

# job-id ranges of the 500k sweep, from lab-notebook 2026-08-15
ARMS = [
    ("uniform",   range(272350, 272375), "uniform replay"),
    ("per",       range(272325, 272350), "PER"),
    ("cce_wmean", range(272300, 272325), "CCE mul, weighted_mean"),
    ("cce_max",   range(272275, 272300), "CCE mul, max"),
]


def pick(jobs):
    scored = []
    for j in jobs:
        c = F.load(str(j), F.RUNS)
        if c is None or len(c[0]) == 0:
            continue
        scored.append((float(np.mean(c[1][-5:])), str(j)))
    if not scored:
        return None, None
    scored.sort()
    return scored[-1] if SEED_PICK == "best" else scored[len(scored) // 2]


def load_policy(job):
    with open(os.path.join(F.RUNS, str(job), "last.pkl"), "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    net = _QNetwork(obs_dim=205, hidden_dim=cfg.get("hidden_dim", 128),
                    n_layers=cfg.get("n_layers", 3), n_actions=15)
    return net, jax.tree.map(jnp.array, ck["params"]), cfg


def build_env(cfg):
    return JaxNavEnv(
        map_id=cfg.get("map_id", "Grid-Rand-Poly"),
        map_size=tuple(cfg.get("map_size", (8, 8))), fill=cfg.get("fill", 0.1),
        goal_radius=cfg.get("goal_radius", 0.8), coll_rew=cfg.get("coll_rew", 0.0),
        max_steps=cfg.get("max_steps", 200), sparse_reward=cfg.get("sparse_reward", True))


def rollout(env, net, params, key):
    raw = env._env
    obs, st = env.reset(key)
    obs_seq, st_seq, rew_seq = [raw.get_obs(st)], [st], [0.0]
    reached = False
    for _ in range(env.max_steps):
        obs, st, r, done, _ = env.step(key, st, int(jnp.argmax(net.apply(params, obs))))
        obs_seq.append(raw.get_obs(st)); st_seq.append(st); rew_seq.append(float(r))
        reached = bool(np.asarray(st.goal_reached).ravel()[0])
        if bool(done):
            break
    return obs_seq, st_seq, rew_seq, reached


def frames(env, o, s, r, title, tmp):
    viz = JaxNavVisualizer(env._env, o, s, r, title_text=title,
                           plot_lidar=False, plot_reward=False)
    viz.animate(save_fname=tmp)
    im = Image.open(tmp)
    fr = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
    im.close(); os.remove(tmp); matplotlib.pyplot.close("all")
    return fr


def main():
    os.makedirs(OUT, exist_ok=True)
    pol = {}
    for arm, jobs, label in ARMS:
        score, job = pick(jobs)
        if job is None:
            print(f"  {arm}: no runs found, skipping", flush=True); continue
        net, params, cfg = load_policy(job)
        pol[arm] = (net, params, build_env(cfg), label, job, score)
        print(f"  {arm:<10} job {job}  ({SEED_PICK} seed, final {score*100:.1f}%)", flush=True)

    keys = list(jax.random.split(jax.random.PRNGKey(ROLLOUT_SEED), N_MAPS))
    summary = {}
    for m, k in enumerate(keys):
        panels, line = [], []
        for arm, (net, params, env, label, job, _) in pol.items():
            o, s, r, ok = rollout(env, net, params, k)
            steps = len(s) - 1
            panels.append(frames(env, o, s, r,
                                 f"{label}   {'GOAL' if ok else 'FAILED'} in {steps} steps",
                                 os.path.join(OUT, f"_t{arm}_{m}.gif")))
            line.append(f"{arm}:{'goal' if ok else 'FAIL'}({steps})")
            summary.setdefault(arm, []).append((ok, steps))
            print(f"   map {m+1}  {arm:<10} {'goal' if ok else 'NO goal':<8} {steps:3d} steps",
                  flush=True)
        n = max(len(p) for p in panels)
        panels = [p + [p[-1]] * (n - len(p)) for p in panels]   # hold finished arms
        w, h = panels[0][0].size
        out = []
        for i in range(n):
            canvas = Image.new("RGB", (w * len(panels), h), "white")
            for j, p in enumerate(panels):
                canvas.paste(p[i], (j * w, 0))
            out.append(canvas)
        out.extend([out[-1]] * 15)
        path = os.path.join(OUT, f"jaxnav_arms_map{m+1}.gif")
        out[0].save(path, save_all=True, append_images=out[1:], duration=60,
                    loop=0, optimize=True)
        print(f"   wrote {path}  ({len(out)} frames)  |  " + "  ".join(line), flush=True)

    print("\n  === across all maps ===", flush=True)
    for arm, res in summary.items():
        g = sum(1 for ok, _ in res if ok)
        st = [s for ok, s in res if ok]
        print(f"   {arm:<10} {g}/{len(res)} goals"
              + (f"   median {np.median(st):.0f} steps when successful" if st else ""), flush=True)


if __name__ == "__main__":
    main()
