"""Submit DoorKey experiment sweeps to SLURM and save manifest files.

Mirrors agents/frozen_lake/run_experiments.py. Two headline experiments:
  - doorkey_claim1 : slip=0.2 (stochastic) checkpointed runs for the oracle-correlation
                     study (CCE's total-variation signal needs stochastic returns).
  - doorkey_claim2 : slip=0.0 (deterministic) 5-algorithm x 10-seed sample-efficiency sweep.

Usage:
    python -m counterfactual_rl.agents.doorkey.run_experiments doorkey_sanity --dry-run
    python -m counterfactual_rl.agents.doorkey.run_experiments doorkey_claim2 --max-concurrent 10
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import date

from counterfactual_rl.agents.shared.slurm_throttle import wait_for_slot


# ── Experiment definitions ────────────────────────────────────────────────────

# Single plain-DQN sanity run (deterministic) — must reach a high win rate before the
# CCE sweep is worth running (pre-flight checklist).
DOORKEY_SANITY = {
    'name': 'doorkey_sanity',
    'runs': [{'algorithm': 'dqn-uniform', 'seed': 0}],
    'fixed': {
        'layout_name': '6x6', 'slip_prob': 0.0,
        'n_episodes': 15000, 'epsilon_decay_episodes': 7500,
        'vectorized': True,
    },
}

# Small full-pipeline smoke: all 5 algorithms, 1 seed, short — exercises the CCE
# scoring path on GPU quickly.
DOORKEY_SMOKE = {
    'name': 'doorkey_smoke',
    'runs': [
        {'algorithm': 'dqn-uniform'},
        {'algorithm': 'dqn'},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 0.25},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative'},
    ],
    'fixed': {
        'layout_name': '6x6', 'slip_prob': 0.0, 'seed': 0,
        'n_episodes': 1500, 'eval_interval': 150, 'eval_episodes': 50,
        'epsilon_decay_episodes': 800, 'score_interval': 50,
        'consequence_metric': 'total_variation', 'mu': 0.25,
        'vectorized': True, 'cf_horizon': 60,
    },
}

# Claim 1 — checkpointed DQN+PER runs on the STOCHASTIC DoorKey (slip=0.2) so the CCE
# total-variation signal is non-degenerate. Post-hoc analysis picks untrained/mid/trained.
DOORKEY_CLAIM1 = {
    'name': 'doorkey_claim1',
    'env_key': 'doorkey',
    'runs': [{'algorithm': 'dqn', 'seed': s} for s in range(3)],
    'fixed': {
        'layout_name': '6x6', 'slip_prob': 0.2,
        'n_episodes': 15000, 'epsilon_decay_episodes': 7500,
        'vectorized': True, 'n_checkpoints': 100,
    },
}

# Claim 2 — 5 algorithms x 10 seeds on the DETERMINISTIC DoorKey (slip=0.0), where CCE
# gets its cleanest sample-efficiency win (mirrors FrozenLake-deterministic).
DOORKEY_CLAIM2 = {
    'name': 'doorkey_claim2',
    'threshold': None,      # run the Claim-2 analysis manually after the sweep
    'env_key': 'doorkey',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                               'seed': s} for s in range(10)],
        *[{'algorithm': 'dqn',                                                       'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0, 'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',             'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',       'seed': s} for s in range(10)],
    ],
    'fixed': {
        'layout_name': '6x6', 'slip_prob': 0.0,
        'n_episodes': 15000, 'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 7500, 'score_interval': 100,
        'vectorized': True, 'cf_horizon': 60,
        'early_stop_win_rate': 0.99,
    },
}

EXPERIMENTS = {
    'doorkey_sanity': DOORKEY_SANITY,
    'doorkey_smoke': DOORKEY_SMOKE,
    'doorkey_claim1': DOORKEY_CLAIM1,
    'doorkey_claim2': DOORKEY_CLAIM2,
}


def generate_runs(experiment):
    fixed = experiment.get('fixed', {})
    result = []
    for run in experiment.get('runs', []):
        overrides = dict(fixed)
        overrides.update(run)
        result.append(overrides)
    return result


# ── Submission ────────────────────────────────────────────────────────────────

def submit_experiment(experiment_name, dry_run=False, max_concurrent=None):
    if experiment_name not in EXPERIMENTS:
        print(f"Error: unknown experiment '{experiment_name}'")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    experiment = EXPERIMENTS[experiment_name]
    runs = generate_runs(experiment)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'train_doorkey_dqn.sh')

    print(f"Experiment: {experiment_name}  ({len(runs)} runs)\n")

    if dry_run:
        for i, overrides in enumerate(runs):
            print(f"  [{i+1:3d}] {overrides}")
        print(f"\n{len(runs)} jobs (dry run — nothing submitted)")
        return

    manifest = {}
    date_str = date.today().isoformat()
    month_str = date_str[:7]
    exp_name = f"{experiment_name}_{date_str}"
    exp_dir = os.path.join(script_dir, 'experiments', month_str, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    for i, overrides in enumerate(runs):
        if max_concurrent is not None:
            wait_for_slot(max_concurrent)
        encoded = base64.b64encode(json.dumps(overrides).encode()).decode()
        cmd = ['sbatch', f'--export=CONFIG_OVERRIDES_B64={encoded}', script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [{i+1:3d}] FAILED: {result.stderr.strip()}")
            continue
        job_id = result.stdout.strip().split()[-1]
        manifest[job_id] = overrides
        print(f"  [{i+1:3d}] Job {job_id}: {overrides}")

    manifest_path = os.path.join(exp_dir, f"{exp_name}.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(manifest)}/{len(runs)} jobs submitted")
    print(f"Manifest: {manifest_path}")
    env_key = experiment.get('env_key')
    if env_key:
        print(f"\nAfter completion, run Claim-2 analysis with:")
        print(f"  python -m counterfactual_rl.analysis.claim2.run_analysis "
              f"--manifest {manifest_path} --env {env_key}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help=f'One of: {", ".join(EXPERIMENTS.keys())}')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-concurrent', type=int, default=None,
                        help='Max jobs in squeue at once (default: no limit)')
    args = parser.parse_args()
    submit_experiment(args.experiment, dry_run=args.dry_run, max_concurrent=args.max_concurrent)
