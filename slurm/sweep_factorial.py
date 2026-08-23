"""JaxNav 6-cell factorial: does CCE beat PER, and does it depend on clutter and size?

    CELLS               fill 0.1     fill 0.3     fill 0.5
      8x8                cell 1     > cell 2 <     cell 3
      11x11              cell 4       cell 5       cell 6

Size x fill are CROSSED, not moved together, so each can be attributed and their
interaction is measurable. That is deliberate: the 2026-08-06 claim "CCE wins
with obstacles, loses without" compared 6x6/fill=0.0 against 8x8/fill=0.1 --
obstacles and size changed at once, and the two conditions overlapped on a third
of episodes because `fill` is a ceiling on a near-uniform draw, not a count
(see the 2026-08-19 LOG). A factorial is the fix for that.

ARMS are the SHIPPED configuration -- the paper's own -- not the ESS-matched
variant built on 2026-08-19. This answers "does the method we publish transfer
across environments", which requires testing what we ship. Whether a win comes
from the CCE *signal* or from CCE's sampler being sharper than PER's is a
separate question, and belongs in one control figure on the headline
environment, not in this grid.

READ THE RESULT AS SIX MARGINS, NOT SIX WIN RATES. `max_steps` is fixed at 200
for every cell, so 11x11 is genuinely harder -- less time to cross a bigger
world. Within a cell every arm faces the same limit, so the CCE-vs-PER margin is
clean; across cells the absolute win rates are not comparable.

NOT COMPARABLE TO ANY PRIOR RUN. Every existing JaxNav and FrozenLake run used
cf_n_rollouts=20 and cf_horizon=20. This uses 40 and 60. That is a deliberate
new baseline: at 20 rollouts and a ~12% goal-reach rate, the standard error on
the difference between two actions is ~0.10 -- the same size as the differences
CCE is trying to rank, i.e. it was largely ranking noise.

    python slurm/sweep_factorial.py --dry-run
    python slurm/sweep_factorial.py --cells 8x8_f03            # the canary
    python slurm/sweep_factorial.py --cells 8x8_f01 11x11_f01 ...   # the rest
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
SBATCH = os.path.join(WT, "slurm", "sweep_job.sbatch")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_factorial.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")
MAX_CONCURRENT = 20

N_SEEDS = 5
TIME_LIMIT = "2-00:00:00"      # 48h. Measured 232 s/1k episodes at 11x11 with
                               # interval=500; interval=250 roughly doubles the
                               # scoring passes, so 250k projects to ~16-25h. The
                               # old 20h limit would have killed runs AT the wall,
                               # wasting the full 20h. teaching allows 7 days.
EXCLUDE = "dh-node12"          # confirmed bad node, see lab-notebook 2026-08-13
CANARY = "8x8_f03"

CELLS = {
    "8x8_f01":   dict(map_size=[8, 8],   fill=0.1),
    "8x8_f03":   dict(map_size=[8, 8],   fill=0.3),
    "8x8_f05":   dict(map_size=[8, 8],   fill=0.5),
    "11x11_f01": dict(map_size=[11, 11], fill=0.1),
    "11x11_f03": dict(map_size=[11, 11], fill=0.3),
    "11x11_f05": dict(map_size=[11, 11], fill=0.5),
}

# Both additive arms use weighted_mean so that arm3-vs-arm5 isolates the MIXING
# rule and arm3-vs-arm4 isolates the AGGREGATION. One variable per comparison.
# cce_only is additive with mu=1.0, i.e. 1*p_c + 0*p_delta -- pure CCE, no TD.
ARMS = {
    "uniform":   dict(algorithm="dqn-uniform"),
    "per":       dict(algorithm="dqn"),
    "cce_wmean": dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                      mu_c=1.0, mu_delta=1.0, consequence_aggregation="weighted_mean"),
    "cce_max":   dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
                      mu_c=1.0, mu_delta=1.0, consequence_aggregation="max"),
    "cce_add":   dict(algorithm="consequence-dqn", priority_mixing="additive",
                      mu=0.25, consequence_aggregation="weighted_mean"),
    "cce_only":  dict(algorithm="consequence-dqn", priority_mixing="additive",
                      mu=1.0, consequence_aggregation="weighted_mean"),
}

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly",
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    # deliberately NOT scaled with the budget, matching every prior JaxNav sweep
    epsilon_decay_episodes=62500,
    n_episodes=250000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,        # disabled, as in every prior sweep
    # Scoring budget. Memory limits ONE pass (n_score x 15 actions x rollouts
    # must stay under ~38,400 parallel envs on a T4 -- 128x15x40 = 76,800 OOMs,
    # measured). score_interval limits how OFTEN, and costs wall-clock rather
    # than memory. So n_score is halved to afford 40 rollouts, and the interval
    # is halved to put coverage back where it was:
    #     2.05 fires/chunk x 64 scored = 131 per 8,192 added = 1.6%, unchanged
    # Rollouts buy precision, which is where the noise is here: only ~12% of
    # rollouts from a buffer state reach the goal, so at 20 rollouts the SE on
    # the difference between two actions is ~0.10 -- the same size as the
    # differences CCE is trying to rank.
    n_score_sample=64,
    score_interval=250,
    cf_rollout_temperature=0.5,
    cf_horizon=60,
    cf_n_rollouts=40,
)

# rough per-run hours, from measured 500k runtimes halved to 250k, x2 for the
# doubled rollout count, x1.4 for the larger map
COST = {"uniform": 1.0, "per": 1.7, "cce_wmean": 8.5,
        "cce_max": 8.5, "cce_add": 8.5, "cce_only": 8.5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be submitted, submit nothing")
    p.add_argument("--cells", nargs="*", default=None,
                   help=f"subset of cells (default: all). Canary is {CANARY}.")
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                   help="SLURM array throttle (%%N). Enforced by the scheduler, "
                        "so it survives this process exiting.")
    p.add_argument("--tag", default="factorial",
                   help="name for the configs file, so reruns do not clobber each other")
    p.add_argument("--partition", default="teaching")
    p.add_argument("--gres", default="gpu:t4:1")
    args = p.parse_args()

    # Hardware is assigned per CELL, never per arm. Same-seed runs do not
    # reproduce on this cluster (GPU float nondeterminism, lab-notebook
    # 2026-08-15), so splitting arms of one cell across GPU types would make
    # hardware a confound in exactly the margin being measured. Cross-cell
    # absolute rates are already not comparable (fixed max_steps), so the cell
    # boundary is the free place to split.

    cells = args.cells if args.cells else list(CELLS)
    arms = args.arms if args.arms else list(ARMS)
    for c in cells:
        if c not in CELLS:
            p.error(f"unknown cell {c!r}; choose from {list(CELLS)}")
    for a in arms:
        if a not in ARMS:
            p.error(f"unknown arm {a!r}; choose from {list(ARMS)}")

    # merge into the existing manifest so the canary and the later cells end up
    # as ONE object the analysis can read as a single factorial
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
                print(f"   {arm:<10} x{args.seeds} seeds   ~{cell_hours:5.1f} GPU-h"
                      f"   {ARMS[arm].get('priority_mixing','-'):<15}"
                      f"{ARMS[arm].get('consequence_aggregation','-')}")
            for seed in range(args.seeds):
                key = f"{cell}/{arm}/{seed}"
                if key in manifest:
                    print(f"   SKIP {key} (already submitted as {manifest[key]['job_id']})")
                    continue
                cfg = {**BASE, **CELLS[cell], **ARMS[arm], "seed": seed}
                pending.append((key, cell, arm, seed, cfg))

    n = len(pending)
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}{n} jobs "
          f"({len(cells)} cells x {len(arms)} arms x {args.seeds} seeds)")
    print(f"  estimated {hours:.0f} GPU-h  ->  ~{hours/14:.0f} h wall-clock at 14 slots")
    print(f"  submitted as ONE array, throttled to {args.max_concurrent} concurrent")
    if args.dry_run:
        print("  Nothing submitted. Drop --dry-run to launch.")
        if CANARY in cells and len(cells) > 1:
            print(f"  TIP: run the canary alone first --  --cells {CANARY}")
        return
    if not n:
        print("  nothing new to submit.")
        return

    # one base64 config per line; array task i reads line i+1
    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfg_path = os.path.join(CONFIG_DIR, f"{args.tag}.txt")
    with open(cfg_path, "w") as fh:
        for _, _, _, _, cfg in pending:
            fh.write(base64.b64encode(json.dumps(cfg).encode()).decode() + "\n")

    cmd = ["sbatch", "--parsable",
           f"--array=0-{n-1}%{args.max_concurrent}",
           f"--time={TIME_LIMIT}",
           f"--partition={args.partition}", f"--gres={args.gres}",
           f"--export=ALL,CONFIG_FILE={cfg_path}", ARRAY_SBATCH]
    if args.partition == "teaching":
        cmd.insert(-2, f"--exclude={EXCLUDE}")   # bad node is teaching-only
    out = subprocess.run(cmd, capture_output=True, text=True)
    array_id = out.stdout.strip()
    if not array_id:
        print(f"  SUBMIT FAILED: {out.stderr.strip()}")
        return

    for i, (key, cell, arm, seed, cfg) in enumerate(pending):
        # cfg already carries `seed`; passing it again as a keyword collides
        manifest[key] = {"job_id": f"{array_id}_{i}", "array_id": array_id,
                         "task": i, "partition": args.partition,
                         "gres": args.gres, "cell": cell, "arm": arm, **cfg}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  array {array_id}  tasks 0-{n-1}  ({args.max_concurrent} at a time)")
    print(f"  partition {args.partition}  gres {args.gres}")
    print(f"  configs:  {cfg_path}")
    print(f"  manifest: {MANIFEST}  ({len(manifest)} runs total)")


if __name__ == "__main__":
    main()
