"""Quick, cheap re-check of the holes map with the weighted_mean aggregation bug
fixed (see repo issue #3 -- consequence_dqn.py was silently falling back to
'max' instead of the requested 'weighted_mean'). Deliberately SHORT (12k
episodes, half the original 3-seed/24k quick test) to get a fast read on
whether the fix changes CCE's behavior before committing to a long run.
No early-stop (kept simple after the plateau-stop confusion in v3/v4).
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
    early_stop_patience=100000,  # disabled -- run the full budget, keep it simple
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
            ["sbatch", "--parsable", "--time=01:00:00",
             f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
        )
        jid = out.decode().strip()
        manifest[jid] = cfg
        print(f"{arm:8s} seed={seed}  job={jid}")

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav", "experiments", "holes_fixed_agg")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
