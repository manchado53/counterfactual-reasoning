"""ESS-matched balance sweep on DETERMINISTIC FrozenLake -- the paper's headline result.

The paper reports CCE+TD(mul) 80% of seeds solved vs PER 48% on FL-det, and fixes
beta=0.25, mu_C=mu_delta=1 without measuring what those settings do to the SAMPLER.
They do a lot: at a common exponent the CCE-driven and TD-driven priorities land at
different concentrations (measured ess_frac ~0.47 vs ~0.87), because CCE scores are
far more skewed than TD errors. So the paper's headline comparison varies the signal
AND the concentration, and a reviewer can reasonably ask which one produced the win.

This sweep holds concentration fixed (ess_frac = TARGET_ESS, solved per arm by the
buffer) and varies ONLY `cce_balance` -- the share of that concentration driven by the
CCE score rather than the TD error. balance 0.0 is exactly PER at matched sharpness.

    if the win SURVIVES  -> the headline gets much stronger; the obvious reviewer
                            objection is answered with a measured control.
    if the win VANISHES  -> the headline was concentration, not consequence. Better
                            to find that ourselves than in review.

Config is CLAIM2_NO_SLIP verbatim (the runs behind the paper figure), except that
priority mixing is driven by cce_balance/target_ess_frac instead of a fixed mu.

    python slurm/sweep_fl_balance_ess.py --dry-run
"""
import argparse
import base64
import json
import os
import subprocess

WT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer"
SBATCH = os.path.join(WT, "slurm", "fl_sweep_job.sbatch")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/frozen_lake/data/manifest_fl_balance_ess.json")

N_SEEDS = 25                   # matches the paper's FL-det sample size
TIME_LIMIT = "03:00:00"        # median FL run 18 min, p90 57 min, max 96 min
EXCLUDE = "dh-node12,dh-node16,dh-node17,dh-node18"   # node12 SIGKILLed ~7% of a past sweep
TARGET_ESS = 0.6
BALANCES = [0.0, 0.25, 0.5, 0.75, 1.0]

BASE = dict(
    map_name="8x8",
    is_slippery=False,          # deterministic: the paper's headline setting
    n_episodes=15000,
    consequence_metric="total_variation",
    epsilon_decay_episodes=7500,
    score_interval=100,
    vectorized=True,
    cf_horizon=200,
    early_stop_win_rate=0.95,
    algorithm="consequence-dqn",
    priority_mixing="multiplicative",
    target_ess_frac=TARGET_ESS,
)

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true")
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
                print(f"\nbalance={bal}  ({100*bal:.0f}% CCE / {100*(1-bal):.0f}% TD)"
                      f"{'   == PER at matched sharpness' if bal == 0.0 else ''}")
                print(f"   ess={TARGET_ESS} eps={cfg['n_episodes']} map={cfg['map_name']} "
                      f"slippery={cfg['is_slippery']} x {args.seeds} seeds")
            continue
        jid = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        manifest[f"bal{bal}_seed{seed}"] = dict(job_id=jid, cce_balance=bal, seed=seed, **BASE)
        print(f"balance={bal:<5} seed={seed:2d}  job={jid}")

if args.dry_run:
    print(f"\nDRY RUN: {n} jobs ({len(balances)} balances x {args.seeds} seeds). Nothing submitted.")
    print(f"  est. wall-clock at ~43 free slots, 18 min median: ~{n*18/43/60:.1f} h "
          f"(worst case at 57 min p90: ~{n*57/43/60:.1f} h)")
else:
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{n} jobs submitted. Manifest: {MANIFEST}")
