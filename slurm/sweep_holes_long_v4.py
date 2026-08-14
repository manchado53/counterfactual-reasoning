"""Holes map, 96k episodes -- SAME config as v3, but with the plateau early-stop
disabled (early_stop_patience set above the max possible number of eval
checkpoints, so it can never trigger). v3's plateau stop fired on every single
seed right around when exploration ended (epsilon_decay_episodes=40000), which
looked like premature cutoff rather than a genuine plateau -- this run removes
that variable and just runs the full 96k episodes, same as the uncapped 48k
run that gave the most trustworthy result so far (PER 63.7% vs CCE 51.0%).
"""
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
N_SEEDS = 5

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.1,
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=40000, n_episodes=96000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    # Disable the plateau early-stop: patience (in eval checkpoints) set above
    # the max possible count (96000/250=384), so it can never fire.
    early_stop_patience=100000,
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
        # 96k with no early stop: worst case ~2x the 48k run's longest job (~1h21m) -> ~2h40m.
        out = subprocess.check_output(
            ["sbatch", "--parsable", "--time=03:30:00",
             f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
        )
        jid = out.decode().strip()
        manifest[jid] = cfg
        print(f"{arm:8s} seed={seed}  job={jid}")

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav", "experiments", "holes_long_v4")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
