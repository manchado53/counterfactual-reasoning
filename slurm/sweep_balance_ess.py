"""ESS-matched balance sweep: does the CCE signal beat the TD signal at equal concentration?

Every previous CCE-vs-PER comparison confounded two things. The arms differed in
WHICH signal drove replay priority, but they also differed in HOW CONCENTRATED
the resulting sampler was -- measured ess_frac 0.87 for pure TD against 0.47 for
pure CCE at a common exponent, because CCE scores are far more skewed than TD
errors. A win could therefore have been sharpness rather than signal quality.

Here the buffer solves for the exponent scale that holds ess_frac at
`TARGET_ESS` for every arm (see ConsequenceReplayBuffer._solve_ess_exponents),
so the only thing varying across arms is `cce_balance`: the fraction of the
(fixed) concentration that comes from the CCE score rather than the TD error.

    cce_balance = 0.0   pure TD    -- PER, at matched concentration
    cce_balance = 1.0   pure CCE

Read the result as a dose-response across the five points, not as five pairwise
tests: a monotone trend across the axis is the claim, and it has more power at
8 seeds than any single comparison would.

Realized concentration is logged per eval to `ess.jsonl` in each run dir, so the
"matched" claim is verifiable after the fact rather than assumed -- including
the `ess_k_saturated` flag, which marks any eval where the target was
unreachable because the driving signal was degenerate.

Budget: 5 arms x 8 seeds = 40 runs at 250k episodes (~5 h each).

    python slurm/sweep_balance_ess.py --dry-run
    python slurm/sweep_balance_ess.py
"""
import argparse
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_balance_ess.json")

N_SEEDS = 8
TIME_LIMIT = "10:00:00"
EXCLUDE = "dh-node12"          # confirmed bad node, see lab-notebook 2026-08-13
TARGET_ESS = 0.6               # mid-range: 0.9 is near-uniform, 0.1 over-concentrates
BALANCES = [0.0, 0.25, 0.5, 0.75, 1.0]

# Matches sweep_holes_25seed_500k.py so this sweep is comparable to the 25-seed
# data already on disk. epsilon_decay stays at 62500 -- deliberately NOT scaled
# with the budget there, and changing it here would confound the comparison.
BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.1,
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=62500,
    n_episodes=250000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,
    algorithm="consequence-dqn",
    priority_mixing="multiplicative",
    score_interval=500, cf_rollout_temperature=0.5,
    cf_horizon=20, cf_n_rollouts=20,
    # weighted_mean, not max: it is monotonic AND linear, so Theorem 1's
    # precondition actually holds, and it was best on every reliability measure
    # at 150k. Choosing `max` because it scored higher would be picking on the
    # outcome.
    consequence_aggregation="weighted_mean",
    target_ess_frac=TARGET_ESS,
)

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true", help="print what would be submitted, submit nothing")
p.add_argument("--seeds", type=int, default=N_SEEDS)
p.add_argument("--balances", type=float, nargs="*", default=None)
args = p.parse_args()

balances = args.balances if args.balances is not None else BALANCES
manifest, n = {}, 0
for bal in balances:
    for seed in range(args.seeds):
        cfg = {**BASE, "cce_balance": bal, "seed": seed}
        b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
        cmd = ["sbatch", "--parsable", f"--time={TIME_LIMIT}", f"--exclude={EXCLUDE}",
               f"--export=ALL,CONFIG_OVERRIDES_B64={b64}", SBATCH]
        n += 1
        if args.dry_run:
            if seed == 0:
                print(f"\nbalance={bal}  ({100*bal:.0f}% CCE / {100*(1-bal):.0f}% TD)")
                print(f"   target_ess_frac={cfg['target_ess_frac']}  episodes={cfg['n_episodes']}"
                      f"  agg={cfg['consequence_aggregation']}")
                print(f"   sbatch --time={TIME_LIMIT} --exclude={EXCLUDE} {SBATCH}")
            continue
        jid = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        manifest[f"bal{bal}_seed{seed}"] = dict(job_id=jid, cce_balance=bal, seed=seed, **BASE)
        print(f"balance={bal:<5} seed={seed:2d}  job={jid}")

if args.dry_run:
    print(f"\nDRY RUN: {n} jobs ({len(balances)} balances x {args.seeds} seeds). Nothing submitted.")
else:
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{n} jobs submitted. Manifest: {MANIFEST}")
