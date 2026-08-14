"""Control arm for the aggregation-fix test: CCE with aggregation explicitly
set to 'max' (reproducing the OLD bug's actual behavior on purpose), same
seeds/budget/map as the already-running weighted_mean and PER jobs
(sweep_holes_holes_fixed_agg.py) -- so all three arms are a clean, matched
A/B/C comparison: same seed -> {max, weighted_mean, PER}.
"""
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
N_SEEDS = 3

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.1,
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=5000, n_episodes=12000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,  # disabled, matches the other two arms
    algorithm="consequence-dqn", priority_mixing="multiplicative",
    score_interval=500, cf_rollout_temperature=0.5, cf_horizon=20, cf_n_rollouts=20,
    consequence_aggregation="max",  # <-- the only thing that differs from the fixed-agg run
)

manifest = {}
for seed in range(N_SEEDS):
    cfg = {**BASE, "seed": seed}
    b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    out = subprocess.check_output(
        ["sbatch", "--parsable", "--time=01:00:00",
         f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
    )
    jid = out.decode().strip()
    manifest[jid] = cfg
    print(f"cce-max  seed={seed}  job={jid}")

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav", "experiments", "holes_max_agg_control")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
