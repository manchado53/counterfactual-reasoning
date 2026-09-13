"""JaxNav with the PUBLISHED reward: uniform, PER and greedy_cov_wm.

WHY. Our JaxNav reward lets an agent wander for free. Under crash = -1 (arrays 275761/275791)
uniform learned to hide: timeouts 29% -> 57%, wins 59% -> 35%. The published setup (No Regrets,
Rutherford et al., NeurIPS 2024; sampling-for-learnability sfl/train/config/env/jaxnav.yaml)
never makes wandering free. This batch copies THAT REWARD exactly and changes nothing else.

THE REWARD, no code change. With sparse_reward=False our wrapper passes only goal_rew/coll_rew
and JaxNav's constructor defaults supply the rest, which equal the paper config term for term:

    goal_rew +4 (set)   coll_rew -4 (set)   dt_rew -0.01   weight_g +0.25   lidar_rew -0.1
    lidar_thresh 0.1 (fires at thresh + rad = 0.4 m, the paper's D_close)
    weight_w -0.0       rew_lambda 0.5 (team mix; a no-op with one robot)

Verified 2026-09-12: compute_reward is byte-identical between our installed jaxmarl 0.1.0 and
every JaxNav commit from June 2024 to upstream main; step_env / goal check / update_state match
the Dec-2024 version. CPU check: a crash step returns -4.11 (crash -4, wall -0.1, time -0.01),
while the sparse env still returns exactly 0 on every non-goal step.

WHAT STAYS OURS, and must be stated in the paper: discrete 15 actions (paper continuous), goal
radius 0.8 (paper 0.3), max_steps 200 (paper 500), 8x8 fill 0.3 (paper 11x11 fill 0.6), random
start/goal placement (paper grid), DQN with replay (paper PPO). Reward only, by the user's call.

WHAT IT CHANGES FOR CCE, noted once: with dense shaping, per-action return spread includes "how
much closer did this action get", not only a different outcome. That is part of what this batch
measures. win_rate is the goal-reached flag, so it stays readable under the shaped reward.

    arm            config                                          ~h/seed (T4)
    uniform_paper  dqn-uniform                                        1.1
    per_paper      dqn                                                1.7
    cov_wm_paper   greedy_cov_wm, identical except the reward        14.5

Compare only INSIDE this reward cell.

    python slurm/sweep_paper_reward.py --dry-run
    python slurm/sweep_paper_reward.py
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_paper_reward.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")

MAX_CONCURRENT = 15            # the whole batch at once; ~15 GPUs, not the cluster
N_SEEDS = 5
# dh-node12: confirmed bad (lab-notebook 2026-08-13). dh-dgx1-1 / dh-dgx1-2 cannot initialise
# CUDA for this conda env (array 275761: 21/21 dead in 16 s; the other DGX nodes ran fine).
EXCLUDE = "dh-node12,dh-dgx1-1,dh-dgx1-2"
# greedy_cov_wm measured 13.8-15.1 h on T4. 36 h is generous on purpose: a run killed at the
# wall wastes the whole run, and the shaped reward may lengthen episodes.
TIME_LIMIT = "1-12:00:00"

PAPER_REWARD = dict(sparse_reward=False, goal_rew=4.0, coll_rew=-4.0)

# greedy_cov_wm, copied key for key from slurm/sweep_greedy_coverage.py (_G + _H60 + _WM with
# n_score_sample 1,280) so the only difference from that arm is the reward.
_COV_WM = dict(
    algorithm="consequence-dqn",
    priority_mixing="multiplicative", mu_c=1.0, mu_delta=1.0,
    consequence_metric="wasserstein",      # forced by cf_n_rollouts=1
    cf_rollout_temperature=0.0,            # greedy -> R=1 is exact
    cf_n_rollouts=1,
    buffer_capacity=100_000,
    cf_horizon=60, cf_bootstrap=True,
    consequence_aggregation="weighted_mean", cf_action_prob_temperature=0.5,
    n_score_sample=1_280, score_interval=250, cf_gamma=0.99,
)
ARMS = {
    "uniform_paper": {"algorithm": "dqn-uniform", **PAPER_REWARD},
    "per_paper":     {"algorithm": "dqn", **PAPER_REWARD},
    "cov_wm_paper":  {**_COV_WM, **PAPER_REWARD},
}

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.3,
    goal_radius=0.8, max_steps=200,
    epsilon_decay_episodes=62500,
    n_episodes=250000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,        # disabled, as in every prior sweep
    n_buffer_snapshots=8,
)

COST = {"uniform_paper": 1.1, "per_paper": 1.7, "cov_wm_paper": 15.1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--tag", default="paper_reward")
    # untyped gpu:1 lets one array span every partition listed
    p.add_argument("--partition", default="dgx,dgxh100,teaching")
    p.add_argument("--gres", default="gpu:1")
    args = p.parse_args()

    arms = args.arms or list(ARMS)
    for a in arms:
        if a not in ARMS:
            p.error(f"unknown arm {a!r}; choose from {list(ARMS)}")

    manifest = {}
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))
        print(f"merging into existing manifest ({len(manifest)} runs recorded)\n")

    pending, hours = [], 0.0
    print(f"{'arm':<15}{'algo':<17}{'goal':>6}{'crash':>7}{'sparse':>8}{'GPU-h':>8}")
    for arm in arms:
        a = ARMS[arm]
        hours += COST[arm] * args.seeds
        print(f"{arm:<15}{a['algorithm']:<17}{a['goal_rew']:>6.1f}{a['coll_rew']:>7.1f}"
              f"{str(a['sparse_reward']):>8}{COST[arm]*args.seeds:>8.1f}")
        for seed in range(args.seeds):
            key = f"{arm}/{seed}"
            if key in manifest:
                print(f"   SKIP {key} (already submitted as {manifest[key]['job_id']})")
                continue
            pending.append((key, arm, seed, {**BASE, **a, "seed": seed}))

    banned = {"cf_horizon", "cf_n_rollouts", "cf_bootstrap", "score_on_insert",
              "consequence_aggregation", "normalise_priorities", "priority_mixing",
              "n_score_sample", "score_interval"}
    for key, arm, _, cfg in pending:
        # the whole point of the batch: the paper reward on EVERY arm
        for k, v in PAPER_REWARD.items():
            if cfg.get(k) != v:
                raise SystemExit(f"FATAL: {key} has {k}={cfg.get(k)!r}, expected {v!r}")
        if cfg["algorithm"] != "consequence-dqn" and banned & set(cfg):
            raise SystemExit(f"FATAL: baseline {key} carries CCE keys {sorted(banned & set(cfg))}")

    n, c = len(pending), args.max_concurrent
    waves = -(-n // c) if n else 0
    slowest = max((COST[k[1]] for k in pending), default=0)
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}{n} jobs ({len(arms)} arms x {args.seeds} seeds)")
    print(f"  estimated {hours:.0f} GPU-h")
    print(f"  slowest arm ~{slowest:.1f} h  ->  {waves} wave(s) of {c}  =  ~{waves*slowest:.1f} h wall")
    print(f"  time limit per run: {TIME_LIMIT}")
    print("  checked: paper reward on every arm; baselines carry no CCE keys")
    if args.dry_run:
        print("  Nothing submitted. Drop --dry-run to launch.")
        return
    if not n:
        print("  nothing new to submit.")
        return

    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfg_path = os.path.join(CONFIG_DIR, f"{args.tag}.txt")
    with open(cfg_path, "w") as fh:
        for *_, cfg in pending:
            fh.write(base64.b64encode(json.dumps(cfg).encode()).decode() + "\n")

    cmd = ["sbatch", "--parsable", f"--array=0-{n-1}%{c}", f"--time={TIME_LIMIT}",
           f"--partition={args.partition}", f"--gres={args.gres}",
           f"--exclude={EXCLUDE}", f"--export=ALL,CONFIG_FILE={cfg_path}", ARRAY_SBATCH]
    out = subprocess.run(cmd, capture_output=True, text=True)
    array_id = out.stdout.strip()
    if not array_id:
        print(f"  SUBMIT FAILED: {out.stderr.strip()}")
        return

    for i, (key, arm, seed, cfg) in enumerate(pending):
        manifest[key] = {"job_id": f"{array_id}_{i}", "array_id": array_id, "task": i,
                         "partition": args.partition, "gres": args.gres, "arm": arm, **cfg}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"  array {array_id}  tasks 0-{n-1}  ({c} at a time)")
    print(f"  configs:  {cfg_path}")
    print(f"  manifest: {MANIFEST}  ({len(manifest)} runs total)")
    print("\n  REMINDER: run slurm/resolve_manifest.py before any analysis -- "
          "array tasks write to JobIDRaw, not <array>_<task>.")


if __name__ == "__main__":
    main()
