"""JaxNav capacity floor — where does a small buffer break, WITHOUT CCE anywhere?

Every small-buffer run this project has ever done was a CCE run:

    consequence-dqn   500 / 1,000 / 2,500 / 8,192 / 10,000 / 25,000 / 100,000
    dqn-uniform       nothing below 100,000
    dqn (PER)         nothing below 100,000

So the lab-notebook's rule -- "32 runs, all collapse, DO NOT shrink the buffer" -- is
CONFOUNDED. We never asked whether plain replay collapses at 8,192 too. If it does, small
buffers are simply bad on JaxNav and CCE is innocent. If it does not, shrinking hurts only
when you PRIORITISE, which would explain all 32 runs and is a finding in its own right.

There is also a 12x hole in the ladder: we have 8,192 and 100,000 and nothing between.

WHY IT MATTERS NOW. The next CCE design (score-on-insert) reaches 100% coverage at a cost
that does not depend on capacity, so capacity stops buying coverage. What it still buys is
FRESHNESS:

    capacity 100,000  ->  mean staleness 3,125 gradient updates
    capacity  25,000  ->  mean staleness   781   (today's MEASURED value is 1,788)

A smaller buffer would let score-on-insert land at 100% coverage AND fresher scores than we
have today -- but only if learning survives down there. This sweep is the ~90 GPU-h gate on
a ~530 GPU-h decision.

THE LADDER. 8,192 is the hard floor: n_envs x collect_steps = 256 x 32 transitions land in
ONE burst and only then do gradient updates run, so below 8,192 part of every burst is
overwritten before a single gradient step ever reads it. You paid to collect it and threw
it away unread.

    capacity    bursts held    why this rung
       8,192        1.0        the hard floor
      12,288        1.5        first rung above it
      16,384        2.0        power-of-two step
      25,000        3.1        the staleness target for score-on-insert
      50,000        6.1        halfway to today
     100,000       12.2        today's default, re-run IN-SWEEP as the anchor

WHY BOTH ARMS. uniform isolates the batch-diversity problem on its own. PER adds prioritised
sampling, which is what CCE actually does, so it is the arm that can show a
prioritise-x-small-buffer interaction. PER is the weaker baseline here (30.8% vs uniform
62.8% at capacity 100,000) so it has less room to fall -- that is why both run, not PER
alone.

100,000 is included even though 21 uniform and 13 PER runs already exist there, because
those came from other sweeps with other RNG streams. An in-sweep anchor makes the ladder an
internal comparison.

NO CCE KEYS ANYWHERE. That is the entire point -- this is a clean control. cf_*,
score_interval, n_score_sample and priority_mixing are absent from BASE and from every arm.

READ-OUT IS BIMODAL. Report the fraction of seeds reaching the good mode plus the median and
every per-seed value; never a mean (JaxNav's paired seed sd is ~30pp). Compare only at
MATCHED episode counts, and never take a final number from an unfinished run -- the fast
seeds are the good seeds. That trap turned a p=0.844 into a fake p=0.007 this week.

    python slurm/sweep_capacity_floor.py --dry-run
    python slurm/sweep_capacity_floor.py
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_capacity_floor.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")

# slurm_throttle counts ALL of the user's queued jobs, but this sweep uses SLURM's own
# --array=%N cap instead, which counts only this array. Set well above the ~9 jobs already
# running so the ladder does not sit behind them.
MAX_CONCURRENT = 30
N_SEEDS = 5
# uniform measured 1.1 h and PER 1.7 h at capacity 100,000 (timing.jsonl, n=21/n=13), and
# both get CHEAPER as capacity drops -- PER rebuilds its priority array over the whole
# buffer every update, so 8,192 does ~12x less of that work. 8 h is generous on purpose: a
# run killed AT the wall wastes the whole run. Short limits also schedule faster.
TIME_LIMIT = "08:00:00"
EXCLUDE = "dh-node12"          # confirmed bad node, lab-notebook 2026-08-13
DEFAULT_CELL = "8x8_f03"       # env default fill; the cell every recent sweep uses

CELLS = {
    "8x8_f03":   dict(map_size=[8, 8],   fill=0.3),
    "8x8_f01":   dict(map_size=[8, 8],   fill=0.1),
    "8x8_f05":   dict(map_size=[8, 8],   fill=0.5),
}

CAPACITIES = [8_192, 12_288, 16_384, 25_000, 50_000, 100_000]
ALGOS = {"uniform": "dqn-uniform", "per": "dqn"}

ARMS = {f"{name}_{cap}": dict(algorithm=algo, buffer_capacity=cap)
        for name, algo in ALGOS.items() for cap in CAPACITIES}

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly",
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=62500,
    n_episodes=250000,                 # FULL length -- this IS an outcome run
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,        # disabled, as in every prior sweep
)

# PER pays a priority-array rebuild proportional to capacity; uniform pays nothing.
# Linear interpolation off the two measured points at capacity 100,000.
COST = {k: (1.1 if k.startswith("uniform") else 1.1 + 0.6 * v["buffer_capacity"] / 100_000)
        for k, v in ARMS.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be submitted, submit nothing")
    p.add_argument("--cells", nargs="*", default=[DEFAULT_CELL])
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--tag", default="capacity_floor")
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
        if args.dry_run:
            print(f"\n=== cell {cell}  map_size={CELLS[cell]['map_size']} "
                  f"fill={CELLS[cell]['fill']} ===")
            print(f"   {'arm':<18}{'capacity':>9}{'bursts':>8}{'staleness':>11}{'GPU-h':>8}")
        for arm in arms:
            cell_hours = COST[arm] * args.seeds
            hours += cell_hours
            cap = ARMS[arm]["buffer_capacity"]
            if args.dry_run:
                print(f"   {arm:<18}{cap:>9,}{cap/8192:>8.1f}{cap/32:>10,.0f}u"
                      f"{cell_hours:>8.1f}")
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
    waves = -(-n // c) if n else 0
    slowest = max((COST[k[2]] for k in pending), default=0)
    print(f"  slowest arm ~{slowest:.1f} h; {n} runs total")
    print(f"  -> {waves} wave(s) of {c} concurrent  =  ~{waves * slowest:.1f} h wall-clock")
    print(f"  time limit per run: {TIME_LIMIT}")
    # A CCE key reaching this sweep would silently turn a control into a treatment.
    banned = {"cf_horizon", "cf_n_rollouts", "cf_bootstrap", "cf_rollout_temperature",
              "score_interval", "n_score_sample", "priority_mixing", "mu", "mu_c",
              "mu_delta", "consequence_aggregation", "normalise_priorities"}
    for key, _, _, _, cfg in pending:
        bad = banned & set(cfg)
        if bad:
            raise SystemExit(f"FATAL: {key} carries CCE keys {sorted(bad)} -- "
                             "this sweep must be a pure control")
    print("  checked: no CCE keys in any config")
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
