"""25 seeds/arm on the holes map at 500,000 episodes, with a UNIFORM-replay arm.

Two things this run is for.

1. A uniform-replay control. The 150k run found PER collapsing late in 7/25
   seeds against CCE+max 1/25 and CCE+wmean 0/25, but with no uniform arm we
   cannot tell "PER's prioritisation causes the collapse" from "this task
   collapses DQN in general and CCE happens to prevent it". Uniform settles it
   and costs PER-speed wall clock (no counterfactual rollouts).

2. More post-exploration training. The collapses appeared at ep 104k-134k, i.e.
   40-70k episodes after exploration ended. epsilon_decay_episodes stays at
   62,500 rather than scaling with the budget, so this run spends 437,500
   episodes past the end of decay instead of the 87,500 the 150k run had.

Seeds are 0-24, the same as the 96k and 150k runs, and the epsilon schedule is
unchanged, so for per/cce-max/cce-wmean the first 150,000 episodes reproduce
that run exactly -- this is a true continuation, and the seeds that collapsed
there (PER 0,6,7,8,10,14,20) can be followed past the point they were cut off.
The flip side, recorded honestly: the collapse/variance finding was discovered
post-hoc on these same seeds, so this run deepens it but does NOT independently
confirm it. That still needs fresh seeds.

Timing, extrapolated from the 150k run (cost is linear in episodes: q-updates
at 150k over q-updates at 75k = 2.02-2.08):
    PER / uniform  ~3.7 h  (worst seed ~4.3 h)
    CCE each arm   ~10.6 h (worst seed ~13.1 h)
--time=20:00:00 leaves real headroom; the 150k run ran a 4:30 limit against a
3:55 worst case, which was too tight to be comfortable.

    python slurm/sweep_holes_25seed_500k.py --dry-run   # always first
    python slurm/sweep_holes_25seed_500k.py
"""
import argparse
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
N_SEEDS = 25
TIME_LIMIT = "20:00:00"
EXCLUDE = "dh-node12"          # confirmed bad node, see lab-notebook 2026-08-13

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.1,
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=62500,   # deliberately NOT scaled with the budget
    n_episodes=500000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,     # disabled, as in the 96k and 150k runs
)

ARMS = {
    "cce-max":   dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                      score_interval=500, cf_rollout_temperature=0.5,
                      cf_horizon=20, cf_n_rollouts=20, consequence_aggregation="max"),
    "cce-wmean": dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                      score_interval=500, cf_rollout_temperature=0.5,
                      cf_horizon=20, cf_n_rollouts=20, consequence_aggregation="weighted_mean"),
    "per":       dict(algorithm="dqn"),
    "uniform":   dict(algorithm="dqn-uniform"),
}

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true",
               help="print what would be submitted, submit nothing")
args = p.parse_args()

manifest = {}
n = 0
for arm, acfg in ARMS.items():
    for seed in range(N_SEEDS):
        cfg = {**BASE, **acfg, "seed": seed}
        b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
        cmd = ["sbatch", "--parsable", f"--time={TIME_LIMIT}", f"--exclude={EXCLUDE}",
               f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
        n += 1
        if args.dry_run:
            if seed == 0:
                print(f"\n{arm}: {cfg['algorithm']}"
                      + (f" / {cfg.get('consequence_aggregation')}"
                         if "consequence_aggregation" in cfg else "")
                      + f"  x{N_SEEDS} seeds")
                print(f"   episodes={cfg['n_episodes']}  "
                      f"eps_decay={cfg['epsilon_decay_episodes']}  "
                      f"score_interval={cfg.get('score_interval', '-')}")
                print(f"   sbatch --time={TIME_LIMIT} --exclude={EXCLUDE} "
                      f"(config via CONFIG_OVERRIDES_B64, {len(b64)} b64 chars)")
            continue
        jid = subprocess.check_output(cmd).decode().strip()
        manifest[jid] = cfg
        print(f"{arm:10s} seed={seed:2d}  job={jid}")

if args.dry_run:
    print(f"\nDRY RUN: {n} jobs would be submitted "
          f"({len(ARMS)} arms x {N_SEEDS} seeds). Nothing was submitted.")
    raise SystemExit(0)

out_dir = os.path.join(WT, "src", "counterfactual_rl", "agents", "jax_nav",
                       "experiments", "holes_25seed_500k")
os.makedirs(out_dir, exist_ok=True)
mpath = os.path.join(out_dir, "manifest.json")
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{len(manifest)} jobs submitted. Manifest: {mpath}")
print("Mirror it into the committed cache once the runs finish:")
print("  export_cache(<manifest>, '25seed_500k')  in analysis.claim2.jaxnav_holes_figures")
