"""JaxNav: buy replay COVERAGE by selling redundant rollouts. Full plan in
plans/jaxnav-greedy-coverage.md.

Coverage has only ever been raised on JaxNav by SHRINKING the buffer, and all 32 such
runs collapse to ~10% (p=0.001) -- which confounds "more coverage" with "less data".
The coverage question was never answered at full capacity: `sample4x` and `interval4x`
exist only at 25,000 episodes with 2 seeds, inside the epsilon decay.

THE TRICK. Scoring work is n_score x actions x rollouts x horizon. JaxNav's transitions
are deterministic, so at cf_rollout_temperature=0 the 20 rollouts of an action are the
SAME rollout 20 times -- one rollout is EXACT, not an approximation. Spend that 20x on
scoring more transitions instead:

    today (cce_base)     64 x 15 x 20 x 60 = 1,152,000 env-steps per scoring pass
    greedy_cov        1,280 x 15 x  1 x 60 = 1,152,000 env-steps per scoring pass

Same GPU budget. ~20x the coverage. Buffer stays at 100,000.

FORCED CONSEQUENCES OF R=1 -- these are not free choices:
  * consequence_metric MUST be wasserstein. compute_total_variation returns exactly 0.0
    when a sample has len < 2, so TV + R=1 silently zeroes every score and degrades CCE
    to uniform replay with no error message.
  * at cf_rollout_temperature=0 the action-probability softmax collapses to one-hot
    (measured ESS 1.0/14 vs 11.0/14 at T=0.5), so 'weighted_mean' silently becomes
    'max'. The _wm arms pin cf_action_prob_temperature=0.5 so the aggregation stays
    exactly what cce_base does and only the ROLLOUT changes.

BOOTSTRAP ON for the H=60 arms. Matched on mixing+aggregation at fill 0.3 / 250k, the
multiplicative arms collapse (<30% final) in 7/24 seeds without bootstrap and 0/13 with
it (Fisher p=0.038). Means +11.7 / +16.2 pp, both n.s. NOTE a similar JaxNav reliability
claim was retracted 2026-08-15 (lab-notebook:622) -- this is a knob setting, not a result.

greedy_full drops bootstrap because at H=200 = max_steps every rollout terminates, so
there is nothing left to bootstrap from. Its premise is doubted by our own measurement
(goal-reach 8% at H=20, 12% at H=60, FLAT through H=160, lab-notebook:624), so it is
SMOKE-GATED: run --arms-full only if the smoke shows rollout_alive_frac ~0 AND
rollout_goal_frac above the 9.2% measured at H=60.

    python slurm/sweep_greedy_coverage.py --smoke --dry-run
    python slurm/sweep_greedy_coverage.py --smoke
    python slurm/sweep_greedy_coverage.py --dry-run
    python slurm/sweep_greedy_coverage.py
"""
import argparse
import base64
import json
import os
import subprocess

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
ARRAY_SBATCH = os.path.join(WT, "slurm", "array_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_greedy_coverage.json")
SMOKE_MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_greedy_smoke.json")
CONFIG_DIR = os.path.join(WT, "slurm", "configs")

MAX_CONCURRENT = 25
N_SEEDS = 5                    # compute-constrained; see READOUT note below
TIME_LIMIT = "1-00:00:00"
EXCLUDE = "dh-node12"          # confirmed bad node, lab-notebook 2026-08-13
CELL = dict(map_size=[8, 8], fill=0.3)   # fill 0.1 is the near-empty-room artifact;
                                         # lab-notebook:667 forbids running there again

# Every arm is multiplicative CCE at full capacity with a greedy single rollout.
# The arms differ in exactly two knobs: n_score_sample (coverage) and aggregation.
_G = dict(
    algorithm="consequence-dqn",
    priority_mixing="multiplicative", mu_c=1.0, mu_delta=1.0,
    consequence_metric="wasserstein",     # forced by cf_n_rollouts=1
    cf_rollout_temperature=0.0,           # greedy walk -> rollouts are deterministic
    cf_n_rollouts=1,                      # exact, not approximate, in a deterministic env
    buffer_capacity=100_000,              # NOT shrinking it -- that is the whole point
)
_MAX = dict(consequence_aggregation="max")                 # ignores action_probs entirely
_WM = dict(consequence_aggregation="weighted_mean",
           cf_action_prob_temperature=0.5)                 # pin weights to cce_base's

_H60 = dict(cf_horizon=60, cf_bootstrap=True)
_H200 = dict(cf_horizon=200, cf_bootstrap=False)           # H = max_steps -> always terminates

ARMS = {
    # coverage 0.86% (today's) -- the reference the coverage arms are measured against
    "greedy_base_max": {**_G, **_H60, **_MAX, "n_score_sample": 64},
    "greedy_base_wm":  {**_G, **_H60, **_WM,  "n_score_sample": 64},
    # coverage ~17% -- differs from the reference by ONE key, n_score_sample
    "greedy_cov_max":  {**_G, **_H60, **_MAX, "n_score_sample": 1_280},
    "greedy_cov_wm":   {**_G, **_H60, **_WM,  "n_score_sample": 1_280},
    # Jeremy's lever 2: reach the SAME coverage by scoring OFTEN instead of BIG.
    # 128/25 == 1280/250 == 5.12 exactly, so coverage matches greedy_cov with clean
    # integers, and the per-chunk transitions scored are 2,624 in both. Same GPU cost:
    # measured on the smoke, scoring time is LINEAR in envs/pass (960 -> 19,200 envs
    # took 17s -> 348s), so 20 small passes and 2 big ones buy the same total.
    # The staleness case for this arm is WEAKER than it first looked: entry lifetime is
    # capacity/n_steps_per_update = 6,250 updates and expected draws per entry is only
    # 0.32, so almost every scored entry is scored ONCE and staleness is set by the draw
    # rate, not by burstiness. Measured: 18x the rate (base -> cov) moved staleness p50
    # only 2,050 -> 1,788. Rearranging the same rate should move it ~interval/2 = 125.
    # Run it anyway: same result as greedy_cov means coverage is what matters and the
    # route does not; a different result means something other than coverage is at play.
    "greedy_freq_max": {**_G, **_H60, **_MAX, "n_score_sample": 128, "score_interval": 25},
    "greedy_freq_wm":  {**_G, **_H60, **_WM,  "n_score_sample": 128, "score_interval": 25},
    # SMOKE-GATED: horizon instead of coverage, no bootstrap needed at H = max_steps.
    # KILLED 2026-09-09: 88-90% of its measured scores are exactly 0.0. Under sparse
    # reward only GOAL pays, so an un-bootstrapped return is binary and ~90% of rollouts
    # return 0 at ANY horizon -- H=200 turns "alive" into "timeout", which pays the same.
    "greedy_full_max": {**_G, **_H200, **_MAX, "n_score_sample": 384},
    "greedy_full_wm":  {**_G, **_H200, **_WM,  "n_score_sample": 384},
}
MAIN_ARMS = ["greedy_base_max", "greedy_base_wm", "greedy_cov_max", "greedy_cov_wm"]
FREQ_ARMS = ["greedy_freq_max", "greedy_freq_wm"]
FULL_ARMS = ["greedy_full_max", "greedy_full_wm"]

BASE = dict(
    scenario=None, map_id="Grid-Rand-Poly",
    goal_radius=0.8, coll_rew=0.0, max_steps=200, sparse_reward=True,
    epsilon_decay_episodes=62500,
    n_episodes=250000,
    eval_interval=250, eval_episodes=100,
    n_envs=256, collect_steps=32, vectorized=True,
    early_stop_patience=100000,
    score_interval=250,
    cf_gamma=0.99,
    n_buffer_snapshots=8,
)

# uniform 1.25h measured; scoring is ~85% of a CCE run and is proportional to
# n_score x rollouts x horizon, normalised to cce_base's 64 x 20 x 60 = 76,800.
_BASE_H, _SCORE_H = 1.25, 7.35


def cost(arm):
    c = ARMS[arm]
    # scoring time measured LINEAR in envs/pass on the smoke, so total scoring cost is
    # (work per pass) x (passes), i.e. proportional to n_score/score_interval.
    work = c["n_score_sample"] * c["cf_n_rollouts"] * c["cf_horizon"]
    freq = BASE["score_interval"] / c.get("score_interval", BASE["score_interval"])
    return _BASE_H + _SCORE_H * freq * work / 76_800.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 seed x 5,000 episodes on every arm, to check the instruments")
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--tag", default=None)
    p.add_argument("--partition", default="teaching")
    p.add_argument("--gres", default="gpu:t4:1")
    args = p.parse_args()

    arms = args.arms if args.arms else (list(ARMS) if args.smoke else MAIN_ARMS)
    for a in arms:
        if a not in ARMS:
            p.error(f"unknown arm {a!r}; choose from {list(ARMS)}")
    seeds = args.seeds if args.seeds is not None else (1 if args.smoke else N_SEEDS)
    manifest_path = SMOKE_MANIFEST if args.smoke else MANIFEST
    tag = args.tag or ("greedy_smoke" if args.smoke else "greedy_coverage")
    episodes = 5_000 if args.smoke else BASE["n_episodes"]
    snaps = 3 if args.smoke else BASE["n_buffer_snapshots"]

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        print(f"merging into existing manifest ({len(manifest)} runs already recorded)\n")

    pending, hours = [], 0.0
    scale = episodes / BASE["n_episodes"]
    print(f"{'arm':<18}{'H':>5}{'boot':>6}{'n_score':>9}  {'agg':<14}{'coverage':>10}{'h/job':>8}")
    print("-" * 72)
    for arm in arms:
        c = ARMS[arm]
        h = cost(arm) * scale
        hours += h * seeds
        cov = 0.86 * (c["n_score_sample"] / 64.0) * (250.0 / c.get("score_interval",
                                                     BASE["score_interval"]))
        print(f"{arm:<18}{c['cf_horizon']:>5}{str(c['cf_bootstrap']):>6}"
              f"{c['n_score_sample']:>9}  {c['consequence_aggregation']:<14}"
              f"{cov:>9.2f}%{h:>8.2f}")
        for seed in range(seeds):
            key = f"{arm}/{seed}"
            if key in manifest:
                print(f"   SKIP {key} (already submitted as {manifest[key]['job_id']})")
                continue
            # arm dict merges AFTER BASE, so an arm may override score_interval
            cfg = {**BASE, **CELL, **c, "seed": seed,
                   "n_episodes": episodes, "n_buffer_snapshots": snaps}
            pending.append((key, arm, seed, cfg))

    n, conc = len(pending), args.max_concurrent
    waves = -(-n // conc) if n else 0
    slowest = max((cost(a) * scale for a in arms), default=0.0)
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}{n} jobs "
          f"({len(arms)} arms x {seeds} seed(s)){'  [SMOKE]' if args.smoke else ''}")
    print(f"  episodes/run {episodes:,}   estimated {hours:.0f} GPU-h")
    print(f"  slowest arm ~{slowest:.2f} h  ->  {waves} wave(s) of {conc}"
          f"  =  ~{waves * slowest:.1f} h wall-clock")
    print(f"  time limit {TIME_LIMIT}   exclude {EXCLUDE}   manifest {os.path.basename(manifest_path)}")
    # coverage is proportional to n_score_sample only while re-scoring collisions stay
    # rare; the 1.6% derivation already reads ~2x high against the measured 0.86%, so
    # the printed percentages are DERIVED. scored_frac in ess.jsonl is the real number.
    print("  NOTE coverage column is DERIVED (scaled from the measured 0.86%);"
          " read scored_frac from ess.jsonl for the truth.")
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

    cmd = ["sbatch", "--parsable", f"--array=0-{n-1}%{conc}",
           f"--time={TIME_LIMIT}", f"--partition={args.partition}", f"--gres={args.gres}",
           f"--export=ALL,CONFIG_FILE={cfg_path}"]
    if args.partition == "teaching":
        cmd.append(f"--exclude={EXCLUDE}")
    cmd.append(ARRAY_SBATCH)
    out = subprocess.run(cmd, capture_output=True, text=True)
    array_id = out.stdout.strip()
    if not array_id:
        print(f"  SUBMIT FAILED: {out.stderr.strip()}")
        return

    for i, (key, arm, seed, cfg) in enumerate(pending):
        manifest[key] = {"job_id": f"{array_id}_{i}", "array_id": array_id, "task": i,
                         "partition": args.partition, "gres": args.gres,
                         "arm": arm, **cfg}
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"  submitted array {array_id} ({n} tasks); manifest -> {manifest_path}")
    print(f"  SLURM array tasks write to JobIDRaw -- run slurm/resolve_manifest.py before analysis.")


if __name__ == "__main__":
    main()
