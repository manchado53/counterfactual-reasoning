"""JaxNav coverage sweep: does a smaller buffer raise replay coverage? (issues #17, #22)

Jeremy's 09/03 action item 1, run as a FALSIFICATION with a pre-registered prediction.

THE PREDICTION, made before the run (simulation in issue #17). Coverage is
scorings-received-per-entry-per-lifetime, and capacity CANCELS out of it:

    coverage = n_score_sample / (score_interval x n_steps_per_update)      <- no capacity

A smaller buffer gives better odds per scoring pass (64 of 10k beats 64 of 100k) but the
entry rides the buffer proportionally fewer insertions, so fewer passes happen while it is
resident. The two factors are the same ratio and cancel.

    buffer_capacity  100,000 -> 0.85%   (control, measured on array 274678)
                      25,000 -> 0.85%
                      10,000 -> 0.88%
                       2,500 -> ~1.1%
    n_score_sample    64 -> 256  ->  ~3.4%   (4x)   <- the lever that should work
    score_interval   250 ->  62  ->  ~3.4%   (4x)   <- same, from the other side

If buffer size moves coverage by more than ~1.2x, the model is WRONG and the notebook
should say so.

WHY THIS IS CHEAP. Coverage has essentially no seed noise -- all 12 CCE runs of array
274678 landed at 0.85% +/- 0.02 -- and it stabilises within ~5k episodes once the buffer
fills. So this needs 25k episodes and 2 seeds, not 250k and 5. ~10 GPU-h instead of ~250.

NOT AN OUTCOME EXPERIMENT. 25k episodes is inside the epsilon decay (62,500), so win rates
here mean nothing and must not be reported. The measured quantity is scored_frac, plus
staleness, which SHOULD fall with a smaller buffer even though coverage does not.

    python slurm/sweep_coverage.py --dry-run
    python slurm/sweep_coverage.py
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_coverage.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")

MAX_CONCURRENT = 30
N_SEEDS = 2          # coverage has ~no seed noise; see module docstring
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
# One CCE configuration (cce_wmean), varied ONLY on the knobs that could move coverage.
# uniform is not included: it never scores, so scored_frac does not exist for it.
_CCE = dict(algorithm="consequence-dqn", priority_mixing="multiplicative",
            mu_c=1.0, mu_delta=1.0, consequence_aggregation="weighted_mean",
            cf_bootstrap=False)
ARMS = {
    # --- the lever Jeremy proposed: buffer size. Predicted to do nothing. ---
    "cap100k": {**_CCE, "buffer_capacity": 100_000},          # control
    "cap25k":  {**_CCE, "buffer_capacity":  25_000},
    "cap10k":  {**_CCE, "buffer_capacity":  10_000},
    "cap2500": {**_CCE, "buffer_capacity":   2_500},
    # --- the levers the arithmetic says DO work, at matched 4x ---
    "sample4x":   {**_CCE, "n_score_sample": 256},            # 4x paint per pass
    "interval4x": {**_CCE, "score_interval":  62},            # 4x as many passes
}

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly",
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=62500,
    n_episodes=25000,     # coverage stabilises by ~5k; NOT an outcome run
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
    # Raw buffer snapshots: 10 per run x ~0.9 MB x 12 CCE-arm runs is ~110 MB, which
    # /home can take. n_checkpoints stays at its default and is a SEPARATE cadence.
    n_buffer_snapshots=8,
)

# measured medians from the factorial's timing.jsonl (n=30/arm), CCE halved for the
# halved rollout count
# 25k episodes = 1/10 of the 250k runs measured at ~8.6 h, so ~0.9 h each. The 4x
# scoring arms pay ~4x the rollout cost, which dominates.
COST = {n: (3.5 if n.endswith("4x") else 0.9) for n in ARMS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be submitted, submit nothing")
    p.add_argument("--cells", nargs="*", default=[DEFAULT_CELL])
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--tag", default="coverage")
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
    # Wall clock is set by the SLOWEST arm, not by an arm-name substring. The
    # inherited version grepped for "cce" in the arm name and reported 0 h here,
    # because this sweep names its arms after the knob being varied.
    n_cce = n
    waves = -(-n // c) if n else 0
    cce_h = max((COST[k[2]] for k in pending), default=0)
    print(f"  slowest arm ~{cce_h:.1f} h; {n} runs total")
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
