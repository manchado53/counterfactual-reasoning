"""JaxNav bootstrap sweep: does fixing the truncated-rollout bug change learning? (#7, #11)

Every CCE arm is run TWICE -- once with `cf_bootstrap` off, once on -- against the
same uniform and PER controls, same seeds, same cell. So the bootstrap effect is a
within-arm paired comparison, not a comparison against a different sweep.

WHY THIS EXISTS. A counterfactual rollout that merely ran out of horizon was scored
as a hard zero, indistinguishable from crashing. Measured effect on the score (jobs
274474 / 274475, probe `analysis/claim2/jaxnav_score_probe.py`):

    cell         states scoring EXACTLY 0      ess_frac @1.43
    8x8_f01           39.1%  ->  15.6%          0.328 -> 0.658
    8x8_f03             (not probed)
    8x8_f05           84.4%  ->  48.4%          0.100 -> 0.270
    11x11_f03         87.5%  ->  32.8%          0.109 -> 0.374

The zero fraction drops a lot. But `ess_frac` ROSE in every cell -- the score got
flatter, not sharper -- and ESS cannot say whether the resulting RANKING is better.
JaxNav has no oracle to settle that. So this sweep settles it the only way left:
by whether it learns better.

NOT COMPARABLE TO THE 6-CELL FACTORIAL. That used cf_n_rollouts=40; this uses 20.
Rollouts were doubled to 40 in the first place because ~90% of rollouts returned
exactly 0, so a per-action mean was dominated by "did one of 40 get lucky". With
bootstrap every rollout carries a value, so the per-rollout variance is far lower
and 20 should suffice -- that is an argument, not a measurement, and halving the
count is what makes 10 arms affordable (CCE 12.8h -> ~6.4h measured median).
Within THIS sweep every arm shares the setting, so the margins are clean.

    python slurm/sweep_bootstrap.py --dry-run
    python slurm/sweep_bootstrap.py                       # 8x8_f03, 5 seeds, 30 at a time
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_bootstrap.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")

MAX_CONCURRENT = 30
N_SEEDS = 5
# Measured CCE median 12.8h at 40 rollouts, worst case 18.1h (factorial timing.jsonl,
# n=30/arm). Halving rollouts projects to ~6.4h median / ~9.1h worst. 24h is generous
# on purpose: a run killed AT the wall wastes the whole run, and teaching allows 7 days.
TIME_LIMIT = "1-00:00:00"
EXCLUDE = "dh-node12"          # confirmed bad node, lab-notebook 2026-08-13
DEFAULT_CELL = "8x8_f03"       # env default fill, real obstacles, and the cell where
                               # PER visibly collapses -- so there is a live effect to
                               # detect. f0.1 is the near-empty-room artifact.

CELLS = {
    "8x8_f01":   dict(map_size=[8, 8],   fill=0.1),
    "8x8_f03":   dict(map_size=[8, 8],   fill=0.3),
    "8x8_f05":   dict(map_size=[8, 8],   fill=0.5),
    "11x11_f01": dict(map_size=[11, 11], fill=0.1),
    "11x11_f03": dict(map_size=[11, 11], fill=0.3),
    "11x11_f05": dict(map_size=[11, 11], fill=0.5),
}

# The four CCE variants from the factorial, unchanged, so the arm definitions stay
# comparable. cce_wmean vs cce_max isolates AGGREGATION; cce_wmean vs cce_add
# isolates MIXING; cce_only is additive with mu=1.0, i.e. pure CCE with no TD.
_CCE = {
    "cce_wmean": dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                      mu_c=1.0, mu_delta=1.0, consequence_aggregation="weighted_mean"),
    "cce_max":   dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                      mu_c=1.0, mu_delta=1.0, consequence_aggregation="max"),
    "cce_add":   dict(algorithm="consequence-dqn", priority_mixing="additive",
                      mu=0.25, consequence_aggregation="weighted_mean"),
    "cce_only":  dict(algorithm="consequence-dqn", priority_mixing="additive",
                      mu=1.0, consequence_aggregation="weighted_mean"),
}

# uniform and PER never compute a consequence score, so cf_bootstrap cannot apply to
# them -- they appear once, as the shared controls for both halves of the sweep.
ARMS = {"uniform": dict(algorithm="dqn-uniform"), "per": dict(algorithm="dqn")}
for _name, _cfg in _CCE.items():
    ARMS[_name] = {**_cfg, "cf_bootstrap": False}
    ARMS[f"{_name}_bs"] = {**_cfg, "cf_bootstrap": True}

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly",
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=62500,
    n_episodes=250000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,        # disabled, as in every prior sweep
    # Coverage is set by n_score_sample x score_interval and is UNCHANGED from the
    # factorial at 1.6% (131 scored per 8,192 added). Halving cf_n_rollouts changes
    # the PRECISION of each scored transition, not how many get scored.
    # Memory: 64 x 15 actions x 20 rollouts = 19,200 parallel envs, half the 38,400
    # the factorial ran and well under the measured T4 ceiling.
    n_score_sample=64,
    score_interval=250,
    cf_rollout_temperature=0.5,
    cf_horizon=60,
    cf_n_rollouts=20,                  # halved vs the factorial -- see module docstring
)

# measured medians from the factorial's timing.jsonl (n=30/arm), CCE halved for the
# halved rollout count
COST = {"uniform": 1.25, "per": 1.72}
for _name in _CCE:
    COST[_name] = COST[f"{_name}_bs"] = 6.4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be submitted, submit nothing")
    p.add_argument("--cells", nargs="*", default=[DEFAULT_CELL])
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--tag", default="bootstrap")
    p.add_argument("--partition", default="teaching")
    p.add_argument("--gres", default="gpu:t4:1")
    args = p.parse_args()

    cells = args.cells
    arms = args.arms if args.arms else list(ARMS)
    for c in cells:
        if c not in CELLS:
            p.error(f"unknown cell {c!r}; choose from {list(CELLS)}")
    for a in arms:
        if a not in ARMS:
            p.error(f"unknown arm {a!r}; choose from {list(ARMS)}")

    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            manifest = json.load(fh)
        print(f"merging into existing manifest ({len(manifest)} runs already recorded)\n")

    pending, hours = [], 0.0
    for cell in cells:
        big = CELLS[cell]["map_size"][0] > 8
        if args.dry_run:
            print(f"\n=== cell {cell}  map_size={CELLS[cell]['map_size']} "
                  f"fill={CELLS[cell]['fill']} ===")
        for arm in arms:
            cell_hours = COST[arm] * args.seeds * (1.4 if big else 1.0)
            hours += cell_hours
            if args.dry_run:
                bs = ARMS[arm].get("cf_bootstrap", "-")
                print(f"   {arm:<14} x{args.seeds}   ~{cell_hours:5.1f} GPU-h"
                      f"   bootstrap={str(bs):<5} "
                      f"{ARMS[arm].get('priority_mixing','-'):<15}"
                      f"{ARMS[arm].get('consequence_aggregation','-')}")
            for seed in range(args.seeds):
                key = f"{cell}/{arm}/{seed}"
                if key in manifest:
                    print(f"   SKIP {key} (already submitted as {manifest[key]['job_id']})")
                    continue
                cfg = {**BASE, **CELLS[cell], **ARMS[arm], "seed": seed}
                pending.append((key, cell, arm, seed, cfg))

    n = len(pending)
    c = args.max_concurrent
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}{n} jobs "
          f"({len(cells)} cell(s) x {len(arms)} arms x {args.seeds} seeds)")
    print(f"  estimated {hours:.0f} GPU-h")
    # Wall clock is wave-based, not GPU-h/slots: the CCE arms are ~5x the baselines
    # and dominate every wave, so the cheap runs ride along in the gaps for free.
    n_cce = sum(1 for k in pending if "cce" in k[2])
    waves = -(-n_cce // c) if n_cce else 0
    cce_h = max((COST[k[2]] for k in pending if "cce" in k[2]), default=0)
    print(f"  {n_cce} of {n} runs are CCE at ~{cce_h:.1f} h each")
    print(f"  -> {waves} wave(s) of {c} concurrent  =  ~{waves * cce_h:.0f} h wall-clock")
    print(f"  time limit per run: {TIME_LIMIT}")
    if args.dry_run:
        print("  Nothing submitted. Drop --dry-run to launch.")
        return
    if not n:
        print("  nothing new to submit.")
        return

    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfg_path = os.path.join(CONFIG_DIR, f"{args.tag}.txt")
    with open(cfg_path, "w") as fh:
        for _, _, _, _, cfg in pending:
            fh.write(base64.b64encode(json.dumps(cfg).encode()).decode() + "\n")

    cmd = ["sbatch", "--parsable",
           f"--array=0-{n-1}%{c}",
           f"--time={TIME_LIMIT}",
           f"--partition={args.partition}", f"--gres={args.gres}",
           f"--export=ALL,CONFIG_FILE={cfg_path}", ARRAY_SBATCH]
    if args.partition == "teaching":
        cmd.insert(-2, f"--exclude={EXCLUDE}")
    out = subprocess.run(cmd, capture_output=True, text=True)
    array_id = out.stdout.strip()
    if not array_id:
        print(f"  SUBMIT FAILED: {out.stderr.strip()}")
        return

    for i, (key, cell, arm, seed, cfg) in enumerate(pending):
        manifest[key] = {"job_id": f"{array_id}_{i}", "array_id": array_id,
                         "task": i, "partition": args.partition,
                         "gres": args.gres, "cell": cell, "arm": arm, **cfg}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  array {array_id}  tasks 0-{n-1}  ({c} at a time)")
    print(f"  configs:  {cfg_path}")
    print(f"  manifest: {MANIFEST}  ({len(manifest)} runs total)")
    print("\n  REMINDER: run slurm/resolve_manifest.py after the array finishes -- "
          "array tasks write to JobIDRaw, not <array>_<task>.")


if __name__ == "__main__":
    main()
