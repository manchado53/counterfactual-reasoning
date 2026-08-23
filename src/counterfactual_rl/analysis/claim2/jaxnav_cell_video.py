"""Side-by-side rollouts of every arm's final policy within one factorial cell.

All six arms play the SAME maps, drawn once up front, advancing in lockstep --
a panel that freezes has already finished, so "who arrives first" is readable
directly.

SEED_PICK defaults to `best` (highest final win rate) because the question here
is "what does the learned policy look like when it works". Use `median` when the
question is about reliability instead -- this project's recurring finding is
that replay strategy shows up in the BAD seeds, which best-seed rendering hides.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src JAX_PLATFORMS=cpu \
      CELL=8x8_f03 N_MAPS=4 SEED_PICK=best \
      python -m counterfactual_rl.analysis.claim2.jaxnav_cell_video <outdir>
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
from jaxmarl.environments.jaxnav import JaxNavVisualizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_factorial.json")
RUNS = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
CELL = os.environ.get("CELL", "8x8_f03")
N_MAPS = int(os.environ.get("N_MAPS", "4"))
ROLLOUT_SEED = int(os.environ.get("ROLLOUT_SEED", "11"))
SEED_PICK = os.environ.get("SEED_PICK", "median")
PANEL_W = int(os.environ.get("PANEL_W", "300"))   # per-panel width in the montage
STRIDE  = int(os.environ.get("STRIDE", "2"))      # keep every Nth frame
HOLD    = int(os.environ.get("HOLD", "8"))        # frames held between maps
ORDER = ["uniform", "per", "cce_wmean", "cce_max", "cce_add", "cce_only"]
LABEL = {"uniform": "uniform", "per": "PER", "cce_wmean": "CCE mul-wmean",
         "cce_max": "CCE mul-max", "cce_add": "CCE add mu=.25", "cce_only": "CCE only mu=1"}


def final_win(run_dir, k=5):
    f = os.path.join(RUNS, str(run_dir), "metrics.log")
    if not os.path.exists(f):
        return None
    vals = []
    for line in open(f):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 5 or p[0] == "episode":
            continue
        vals.append((float(p[0]), float(p[3].rstrip("%")) / 100))
    if not vals or vals[-1][0] < 240_000:      # only fully-trained seeds
        return None
    return float(np.mean([v for _, v in vals[-k:]]))


def pick_per_arm():
    man = json.load(open(MANIFEST))
    byarm = {}
    for rec in man.values():
        if rec["cell"] != CELL or not rec.get("run_dir"):
            continue
        w = final_win(rec["run_dir"])
        if w is None:
            continue
        byarm.setdefault(rec["arm"], []).append((w, rec["run_dir"], rec["seed"]))
    out = {}
    for arm, v in byarm.items():
        v.sort()
        out[arm] = v[-1] if SEED_PICK == "best" else v[len(v) // 2]
    return out


def load_policy(run_dir):
    with open(os.path.join(RUNS, str(run_dir), "last.pkl"), "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    net = _QNetwork(obs_dim=205, hidden_dim=cfg.get("hidden_dim", 128),
                    n_layers=cfg.get("n_layers", 3), n_actions=15)
    env = JaxNavEnv(map_id=cfg["map_id"], map_size=tuple(cfg["map_size"]), fill=cfg["fill"],
                    goal_radius=cfg["goal_radius"], coll_rew=cfg["coll_rew"],
                    max_steps=cfg["max_steps"], sparse_reward=cfg["sparse_reward"])
    return net, jax.tree.map(jnp.array, ck["params"]), env


def rollout(env, net, params, key):
    raw = env._env
    obs, st = env.reset(key)
    o_seq, s_seq, r_seq = [raw.get_obs(st)], [st], [0.0]
    reached = False
    for _ in range(env.max_steps):
        obs, st, r, done, _ = env.step(key, st, int(jnp.argmax(net.apply(params, obs))))
        o_seq.append(raw.get_obs(st)); s_seq.append(st); r_seq.append(float(r))
        reached = bool(np.asarray(st.goal_reached).ravel()[0])
        if bool(done):
            break
    return o_seq, s_seq, r_seq, reached


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
    picks = pick_per_arm()
    if not picks:
        print(f"no completed runs in cell {CELL}", flush=True); return
    arms = [a for a in ORDER if a in picks]
    pol = {}
    print(f"cell {CELL}  ({SEED_PICK} seed per arm)", flush=True)
    for a in arms:
        w, rd, sd = picks[a]
        net, params, env = load_policy(rd)
        pol[a] = (net, params, env, w)
        print(f"  {a:<10} run {rd}  seed {sd}  final {w*100:.1f}%", flush=True)

    keys = list(jax.random.split(jax.random.PRNGKey(ROLLOUT_SEED), N_MAPS))
    tally = {a: [] for a in arms}
    all_frames = []
    for m, k in enumerate(keys):
        panels, line = [], []
        for a in arms:
            net, params, env, w = pol[a]
            o, s_, r, ok = rollout(env, net, params, k)
            steps = len(s_) - 1
            panels.append(frames(env, o, s_, r,
                                 f"{LABEL[a]}  {'GOAL' if ok else 'FAILED'} {steps}",
                                 os.path.join(OUT, f"_c{a}_{m}.gif")))
            tally[a].append((ok, steps)); line.append(f"{a}:{'goal' if ok else 'FAIL'}({steps})")
        n = max(len(p) for p in panels)
        panels = [p + [p[-1]] * (n - len(p)) for p in panels]     # hold finished arms
        w0, h0 = panels[0][0].size
        # downscale + subsample: 6 panels x 10 maps x 200 frames is otherwise ~100 MB
        tw = int(w0 * PANEL_W / w0) if False else PANEL_W
        th = int(h0 * PANEL_W / w0)
        for i in range(0, n, STRIDE):
            cv = Image.new("RGB", (PANEL_W * len(panels), th), "white")
            for j, p in enumerate(panels):
                cv.paste(p[i].resize((PANEL_W, th), Image.LANCZOS), (j * PANEL_W, 0))
            all_frames.append(cv)
        all_frames.extend([all_frames[-1]] * HOLD)                # pause between maps
        print(f"  map {m+1}/{N_MAPS}: " + "  ".join(line), flush=True)

    path = os.path.join(OUT, f"jaxnav_{CELL}_{SEED_PICK}_x{N_MAPS}.gif")
    all_frames[0].save(path, save_all=True, append_images=all_frames[1:],
                       duration=70, loop=0, optimize=True)
    mb = os.path.getsize(path) / 1e6
    print(f"\n  wrote {path}  ({len(all_frames)} frames, {mb:.1f} MB)", flush=True)

    print("  === across all maps ===", flush=True)
    for a in arms:
        g = sum(1 for ok, _ in tally[a] if ok)
        st = [x for ok, x in tally[a] if ok]
        print(f"   {LABEL[a]:<16} {g}/{len(tally[a])} goals"
              + (f"   median {np.median(st):.0f} steps" if st else ""), flush=True)


if __name__ == "__main__":
    main()
