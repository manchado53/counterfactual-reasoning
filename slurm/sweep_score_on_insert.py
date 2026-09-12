"""Score-on-insert at 100% coverage, crossed with the collision reward.

WHY SCORE-ON-INSERT. CCE's replay signal is ~84% fabricated: a slot is scored only if a
random draw lands on it, and over its whole 6,250-update life it gets 0.32 expected draws,
so ~73% of slots die carrying the running mean they inherited on insertion -- and the
sampler cannot tell those apart from real measurements. Raising coverage by drawing MORE
is the coupon-collector trap: 90% coverage costs 1,000,000 rollouts per buffer turnover
(31x today). Visiting each arrival exactly once costs 100,000 (3.1x) and reaches 100%.

WHY THE REWARD ARM, and it may matter more. With coll_rew=0.0 and sparse_reward=True a
crash and a timeout both return 0 -- the same number. So next to a wall, where every action
crashes, the divergence among per-action returns is EXACTLY 0. That is not a CCE bug; it is
CCE correctly reporting that the action choice changed nothing. It explains three findings
at once: 47.8% of map positions scoring exactly 0, CCE correlating +0.688 with
distance-to-obstacle (highest in OPEN space), and 28.5% zeros even with bootstrap on.
coll_rew=-1 makes crashing genuinely worse than wandering, so consequence should appear
exactly where the obstacles are.

    2x2, all weighted_mean + rank, 100% coverage, 5 seeds

                     coll_rew = 0.0        coll_rew = -1.0
        H = 60       soi_h60_r0            soi_h60_rneg
        H = 200      soi_h200_r0           soi_h200_rneg

Plus uniform and PER at coll_rew=-1 only -- that cell has no baselines. coll_rew=0 already
has 26 uniform / 18 PER runs on disk. win_rate stays the goal-reached fraction, so the
read-out is comparable across reward cells even though the training signal is not.

RECORDED DISAGREEMENT, so it is not re-argued later. H=200 measures ~89% zero scores against
H=60's 28.5%, because max_steps=200 means the episode ends BEFORE the horizon and
cf_bootstrap never fires -- there is no truncation for it to repair. I expect the H=200 half
to be 100% coverage of mostly-zeros. It runs anyway: nobody has measured that cell at full
coverage, and the smoke settles it for a few GPU-hours instead of an argument.

SETTLED KNOBS, do not re-open: capacity 100,000 (everything below 50k collapses, array
275562, and capacity buys no coverage above the 8,192 burst anyway) - cf_rollout_temperature
0.0, which makes cf_n_rollouts=1 EXACT rather than approximate - cf_bootstrap True (without
it 82-89% of scores are exactly 0) - wasserstein, because total_variation returns exactly
0.0 at len<2 and R=1 forces it - weighted_mean (max is significantly WORSE than uniform,
p=0.048) - rank normalisation, the only mode with a principled basis (cv 47.4 -> 0.58).

SMOKE FIRST. --smoke runs the four CCE cells at 5,000 episodes, 1 seed, ~8 GPU-h, ~3h wall.
It replaces a 34-152 h/seed EXTRAPOLATION with measured hours, and settles the H=200
zero-fraction. Do not launch the full sweep off the extrapolation -- it is up to 2,000 GPU-h.

    python slurm/sweep_score_on_insert.py --smoke --dry-run
    python slurm/sweep_score_on_insert.py --smoke
    python slurm/sweep_score_on_insert.py --dry-run
    python slurm/sweep_score_on_insert.py
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_score_on_insert.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")

# The user asked not to take the whole cluster. 24 leaves headroom on dgx/dgxh100 and
# touches none of the 80 T4s on teaching. All 20 long CCE runs still start in wave 1;
# the 10 baselines are 1-2 h and trickle through behind them.
MAX_CONCURRENT = 24
N_SEEDS = 5
# dh-node12: confirmed bad, lab-notebook 2026-08-13.
# dh-dgx1-1 / dh-dgx1-2: CANNOT initialise CUDA for this conda env --
#   "CUDA_ERROR_INVALID_VALUE" then "no supported devices found for platform CUDA",
#   within 16 s, 21/21 jobs, while dh-dgx1-3 and dh-dgxh100-1 ran the identical config
#   fine (array 275761, 2026-09-12). Node problem, not a code problem.
EXCLUDE = "dh-node12,dh-dgx1-1,dh-dgx1-2"
# teaching TIMELIMIT is 7 days. H=200 projects to 58-152 h/seed, so 24h would kill it and a
# run killed AT the wall wastes the whole run. The smoke replaces this guess with a measurement.
# dgx / dgxh100 TIMELIMIT is 21 days. The hour figures below are T4-measured; V100 and
# especially H100 are substantially faster, so these are upper bounds twice over. A run
# killed AT the wall wastes the whole run, so the limit is deliberately generous.
TIME_LIMIT = "6-00:00:00"
SMOKE_TIME_LIMIT = "06:00:00"

_CCE = dict(
    algorithm="consequence-dqn",
    score_on_insert=True,              # <- the whole point: kappa = 1 by construction
    cf_score_chunk=1280,               # 1280 x 15 actions = 19,200 envs, the proven T4 width
    cf_rollout_temperature=0.0,        # greedy -> R=1 is EXACT
    cf_n_rollouts=1,
    cf_bootstrap=True,
    consequence_metric="wasserstein",  # total_variation returns exactly 0.0 at len<2
    consequence_aggregation="weighted_mean",
    normalise_priorities="rank",
    priority_mixing="multiplicative", mu_c=1.0, mu_delta=1.0,
    # weighted_mean weights come from a softmax at THIS temperature, decoupled from the
    # rollout temperature on purpose: at T=0 the softmax goes one-hot and weighted_mean
    # silently becomes max, which would make this a disguised aggregation change.
    cf_action_prob_temperature=0.5,
)
_BASE_ALGOS = {"uniform": dict(algorithm="dqn-uniform"), "per": dict(algorithm="dqn")}

ARMS = {}
for _h in (60, 200):
    for _tag, _cr in (("r0", 0.0), ("rneg", -1.0)):
        ARMS[f"soi_h{_h}_{_tag}"] = {**_CCE, "cf_horizon": _h, "coll_rew": _cr}
# baselines ONLY for the new reward cell; coll_rew=0 already has 26 uniform / 18 PER runs
for _name, _cfg in _BASE_ALGOS.items():
    ARMS[f"{_name}_rneg"] = {**_cfg, "coll_rew": -1.0}

SMOKE_ARMS = [k for k in ARMS if k.startswith("soi_")]

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly", map_size=[8, 8], fill=0.3,
    goal_radius=0.8, max_steps=200, sparse_reward=True, goal_rew=1.0,
    epsilon_decay_episodes=62500,
    n_episodes=250000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    buffer_capacity=100000,
    early_stop_patience=100000,        # disabled, as in every prior sweep
    n_buffer_snapshots=8,
)

# H=60 -> 34-51 h/seed, H=200 -> 58-152 h/seed (see module docstring). Upper bound used, so
# the dry-run states the worst case rather than the hopeful one.
def _cost(arm):
    if not arm.startswith("soi_"):
        return 1.1 if arm.startswith("uniform") else 1.7
    return 51.0 if "h60" in arm else 152.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="4 CCE cells, 5,000 episodes, 1 seed -- measure before committing")
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--tag", default=None)
    # gpu:1 (untyped) is what lets one array span BOTH partitions -- gpu:v100:1 would
    # pin it to dgx and gpu:h100:1 to dgxh100.
    p.add_argument("--partition", default="dgx,dgxh100,teaching")
    p.add_argument("--gres", default="gpu:1")
    args = p.parse_args()

    arms = args.arms or (SMOKE_ARMS if args.smoke else list(ARMS))
    for a in arms:
        if a not in ARMS:
            p.error(f"unknown arm {a!r}; choose from {list(ARMS)}")
    seeds = 1 if args.smoke else args.seeds
    episodes = 5_000 if args.smoke else BASE["n_episodes"]
    tag = args.tag or ("score_on_insert_smoke" if args.smoke else "score_on_insert")
    time_limit = SMOKE_TIME_LIMIT if args.smoke else TIME_LIMIT
    manifest_path = MANIFEST.replace(".json", "_smoke.json") if args.smoke else MANIFEST

    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        print(f"merging into existing manifest ({len(manifest)} runs recorded)\n")

    pending, hours = [], 0.0
    scale = episodes / BASE["n_episodes"]
    print(f"{'arm':<16}{'algo':<17}{'H':>5}{'coll_rew':>10}{'kappa':>8}{'GPU-h':>9}")
    for arm in arms:
        cfg_arm = ARMS[arm]
        h = _cost(arm) * seeds * scale
        hours += h
        print(f"{arm:<16}{cfg_arm['algorithm']:<17}{str(cfg_arm.get('cf_horizon','-')):>5}"
              f"{cfg_arm.get('coll_rew', 0.0):>10.1f}"
              f"{'100%' if cfg_arm.get('score_on_insert') else '-':>8}{h:>9.1f}")
        for seed in range(seeds):
            key = f"{arm}/{seed}"
            if key in manifest:
                print(f"   SKIP {key} (already submitted as {manifest[key]['job_id']})")
                continue
            cfg = {**BASE, **cfg_arm, "n_episodes": episodes, "seed": seed}
            pending.append((key, arm, seed, cfg))

    # A control that silently carries CCE keys is not a control.
    banned = {"cf_horizon", "cf_n_rollouts", "cf_bootstrap", "score_on_insert",
              "consequence_aggregation", "normalise_priorities", "priority_mixing"}
    for key, arm, _, cfg in pending:
        if cfg["algorithm"] != "consequence-dqn":
            bad = banned & set(cfg)
            if bad:
                raise SystemExit(f"FATAL: baseline {key} carries CCE keys {sorted(bad)}")
        elif not cfg.get("score_on_insert"):
            raise SystemExit(f"FATAL: CCE arm {key} has score_on_insert off -- "
                             "this sweep exists to test it")

    n, c = len(pending), args.max_concurrent
    waves = -(-n // c) if n else 0
    slowest = max((_cost(k[1]) * scale for k in pending), default=0)
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}{n} jobs ({len(arms)} arms x {seeds} seeds)"
          f"   {episodes:,} episodes")
    print(f"  estimated {hours:.0f} GPU-h (upper bound)")
    print(f"  slowest arm ~{slowest:.1f} h  ->  {waves} wave(s) of {c}  =  "
          f"~{waves*slowest:.1f} h wall-clock")
    print(f"  time limit per run: {time_limit}")
    print("  checked: baselines carry no CCE keys; every CCE arm has score_on_insert=True")
    if args.dry_run:
        print("  Nothing submitted. Drop --dry-run to launch.")
        return
    if not n:
        print("  nothing new to submit.")
        return

    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfg_path = os.path.join(CONFIG_DIR, f"{tag}.txt")
    with open(cfg_path, "w") as fh:
        for _, _, _, cfg in pending:
            fh.write(base64.b64encode(json.dumps(cfg).encode()).decode() + "\n")

    cmd = ["sbatch", "--parsable", f"--array=0-{n-1}%{c}", f"--time={time_limit}",
           f"--partition={args.partition}", f"--gres={args.gres}",
           f"--export=ALL,CONFIG_FILE={cfg_path}", ARRAY_SBATCH]
    cmd.insert(-2, f"--exclude={EXCLUDE}")
    out = subprocess.run(cmd, capture_output=True, text=True)
    array_id = out.stdout.strip()
    if not array_id:
        print(f"  SUBMIT FAILED: {out.stderr.strip()}")
        return

    for i, (key, arm, seed, cfg) in enumerate(pending):
        manifest[key] = {"job_id": f"{array_id}_{i}", "array_id": array_id, "task": i,
                         "partition": args.partition, "gres": args.gres, "arm": arm, **cfg}
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"  array {array_id}  tasks 0-{n-1}  ({c} at a time)")
    print(f"  configs:  {cfg_path}")
    print(f"  manifest: {manifest_path}  ({len(manifest)} runs total)")
    print("\n  REMINDER: run slurm/resolve_manifest.py before any analysis -- "
          "array tasks write to JobIDRaw, not <array>_<task>.")


if __name__ == "__main__":
    main()
