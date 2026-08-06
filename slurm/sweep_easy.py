"""Option B: 5-seed CCE-mul / PER / uniform sweep on the EASY (obstacle-free) map.

Submits 15 jobs via CONFIG_OVERRIDES_B64 and writes a manifest {job_id: config} that
analysis/claim2/run_analysis consumes.
"""
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
N_SEEDS = 5

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[6, 6], fill=0.0,
    goal_radius=1.0, max_steps=150, sparse_reward=True,
    epsilon_decay_episodes=4000, n_episodes=8000,
    eval_interval=200, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    # stabilized defaults are already in DEFAULT_CONFIG (double_dqn, n_steps_per_update=16, ...)
)

ARMS = {
    "cce-mul": dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                    score_interval=500, cf_rollout_temperature=0.5,
                    cf_horizon=20, cf_n_rollouts=20),
    "per":     dict(algorithm="dqn"),
    "uniform": dict(algorithm="dqn-uniform"),
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

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav", "experiments", "easy_b")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
