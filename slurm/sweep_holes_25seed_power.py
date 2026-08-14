"""Power-analysis-justified rerun: 25 seeds per arm (up from 5), 96k episodes,
holes map. Three arms:
  - CCE + max          (reproduces the old aggregation bug on purpose, as a control)
  - CCE + weighted_mean (the fix -- real action_probs, genuine weighted mean)
  - PER                 (baseline)

Why 25: a Monte Carlo power analysis using the observed 5-seed means/stds
(CCE 70.3% +-8.3%, PER 56.1% +-22.8%) showed 5 seeds/arm gives only ~19%
power to detect this effect size at p<0.05; ~25 seeds/arm is needed to reach
the conventional 80% power bar. PER's wide spread (one seed at 15%, another
at 83%) is what's driving the large N requirement.
"""
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
N_SEEDS = 25

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.1,
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=40000, n_episodes=96000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,  # disabled -- run the full budget, no plateau-stop surprises
)

ARMS = {
    "cce-max":     dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                         score_interval=500, cf_rollout_temperature=0.5,
                         cf_horizon=20, cf_n_rollouts=20,
                         consequence_aggregation="max"),
    "cce-wmean":   dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                         score_interval=500, cf_rollout_temperature=0.5,
                         cf_horizon=20, cf_n_rollouts=20,
                         consequence_aggregation="weighted_mean"),
    "per":         dict(algorithm="dqn"),
}

manifest = {}
for arm, acfg in ARMS.items():
    for seed in range(N_SEEDS):
        cfg = {**BASE, **acfg, "seed": seed}
        b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
        out = subprocess.check_output(
            ["sbatch", "--parsable", "--time=03:30:00",
             f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
        )
        jid = out.decode().strip()
        manifest[jid] = cfg
        print(f"{arm:10s} seed={seed:2d}  job={jid}")

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav", "experiments", "holes_25seed_power")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
