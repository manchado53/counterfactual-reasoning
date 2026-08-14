"""Holes map, 96k episodes (2x the last confirmation run), with real plateau
early-stopping so a converged run doesn't grind through the full budget for
nothing. Follow-up to the 5-seed/48k run, which found both CCE and PER had
mostly flattened by the end (per-seed trend checked directly: most seeds
+-1pp/eval or worse in the final stretch) -- this run doubles the ceiling but
lets each seed stop on its own once it plateaus, rather than assuming 96k is
the right number for everyone.
"""
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
N_SEEDS = 5

# Same holes-map config as v2, 2x episodes, epsilon decay keeps the ~42% ratio
# used in every holes run so far (10000/24000, 20000/48000, now 40000/96000).
BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.1,
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=40000, n_episodes=96000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    # Plateau early-stop: stop once smoothed win-rate hasn't improved by 2pp
    # for 20 evals (5000 episodes) after epsilon has finished decaying.
    early_stop_patience=20, early_stop_min_delta=0.02, early_stop_smooth_window=5,
)

ARMS = {
    "cce-mul": dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                    score_interval=500, cf_rollout_temperature=0.5,
                    cf_horizon=20, cf_n_rollouts=20),
    "per":     dict(algorithm="dqn"),
}

manifest = {}
for arm, acfg in ARMS.items():
    for seed in range(N_SEEDS):
        cfg = {**BASE, **acfg, "seed": seed}
        b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
        # --time override (not editing sweep_job.sbatch's default 03:00:00, which other
        # sweeps also use): 96k episodes without an early stop needs more headroom than the
        # 48k run's longest job (~1h21m) x2 -- round up for safety.
        out = subprocess.check_output(
            ["sbatch", "--parsable", "--time=03:30:00",
             f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
        )
        jid = out.decode().strip()
        manifest[jid] = cfg
        print(f"{arm:8s} seed={seed}  job={jid}")

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav", "experiments", "holes_long_v3")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
