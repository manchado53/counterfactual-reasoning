"""JaxNav instrumentation run: what is the buffer actually DOING? (issues #8, #9)

A MECHANISM run, not an outcome run. n=3 on purpose. JaxNav's paired seed sd is 30pp
and detecting a 5pp effect needs ~290 seed-pairs (lab-notebook 2026-09-03), so no
affordable seed count could show that any arm LEARNS better here, and nothing from
this sweep may be reported as if it could. What 3 seeds can do is show what the
replay buffer contains and what the counterfactual rollouts believe -- neither of
which had ever been recorded before the instruments landed.

WHAT IS NEW SINCE THE BOOTSTRAP SWEEP. Every run now writes:
  ess.jsonl   scored_frac (measured replay COVERAGE), staleness p50/p90, a 100-bin
              log-spaced score histogram every eval, rollout terminal reasons, and
              the double-sampling overlap against a TD-only shadow batch
  eval.jsonl  crash / timeout rates and length-conditional-on-outcome, for EVERY
              arm including uniform and PER
  buffer/     10 npz snapshots of the raw priority arrays per run

ALREADY MEASURED BY THE SMOKE RUN (274677, 5k episodes). Coverage is real and it is
WORSE than the derived number: the notebook's 1.6% is what one transition accumulates
over a full LIFETIME, but the buffer holds entries at every age at once, so what the
sampler actually sees is ~0.9%. The scored fraction ramps with age (0.14% in the
newest tenth of the buffer to 1.84% in the oldest), which is exactly the shape
1 - exp(-lambda*u) predicts. This sweep is what extends that from 5k to 250k episodes.

ARMS. The four CCE arms keep the bootstrap sweep's definitions so the comparisons stay
clean, and every one of them holds cf_bootstrap FIXED except the pair that tests it:
    cce_wmean vs cce_max        isolates AGGREGATION   (both bootstrap OFF)
    cce_wmean vs cce_only       isolates MIXING        (both bootstrap OFF)
    cce_wmean vs cce_wmean_bs   isolates BOOTSTRAP     (aggregation/mixing held)

    python slurm/sweep_instrument.py --dry-run
    python slurm/sweep_instrument.py                   # 8x8_f03, 3 seeds
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_instrument.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")

MAX_CONCURRENT = 30
N_SEEDS = 3          # mechanism run; see module docstring for why not more
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
# cce_add is dropped (cce_only already covers additive mixing at the informative end,
# mu=1.0). Only cce_wmean gets a _bs twin: bootstrap was settled as a null over 20
# seed-matched pairs (Wilcoxon p=0.837), so a second pair here would buy nothing --
# the twin is kept ONLY so this sweep still contains the paired contrast.
for _name in ("cce_wmean", "cce_max", "cce_only"):
    ARMS[_name] = {**_CCE[_name], "cf_bootstrap": False}
ARMS["cce_wmean_bs"] = {**_CCE["cce_wmean"], "cf_bootstrap": True}

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
    # Raw buffer snapshots: 10 per run x ~0.9 MB x 12 CCE-arm runs is ~110 MB, which
    # /home can take. n_checkpoints stays at its default and is a SEPARATE cadence.
    n_buffer_snapshots=10,
)

# measured medians from the factorial's timing.jsonl (n=30/arm), CCE halved for the
# halved rollout count
COST = {"uniform": 1.25, "per": 1.72}
for _name in ARMS:
    if _name.startswith("cce"):
        COST[_name] = 6.4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be submitted, submit nothing")
    p.add_argument("--cells", nargs="*", default=[DEFAULT_CELL])
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--tag", default="instrument")
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
