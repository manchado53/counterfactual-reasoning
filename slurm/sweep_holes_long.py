"""Holes map, 2x episode budget, 3 seeds each for CCE-mul and PER.

Follow-up to the 1-seed holes pilot (271050/271051): that run showed CCE still
climbing at the end (+0.76pp/eval) while PER had plateaued (-0.24pp/eval).
This run gives that trend more time and enough seeds to trust it.
"""
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
N_SEEDS = 3

# Same holes-map config as the 1-seed pilot (271050/271051), just 2x episodes.
BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.1,
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=10000, n_episodes=24000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
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
        out = subprocess.check_output(
            ["sbatch", "--parsable", f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
        )
        jid = out.decode().strip()
        manifest[jid] = cfg
        print(f"{arm:8s} seed={seed}  job={jid}")

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav", "experiments", "holes_long")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
