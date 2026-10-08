"""Submit Connect Four experiment sweeps to SLURM and save manifest files.

Usage:
    python run_experiments.py smoke --dry-run
    python run_experiments.py smoke
    python run_experiments.py claim2_main --dry-run
    python run_experiments.py claim2_main
    python run_experiments.py claim2_main --max-concurrent 8
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

# Smoke test — all 5 algorithms, 1 seed, 20 chunks.
# Use to verify: env works, CCE scores, curves show learning, no OOM.
SMOKE = {
    'name': 'smoke',
    'runs': [
        {'algorithm': 'dqn-uniform'},
        {'algorithm': 'dqn'},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive'},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative'},
    ],
    'fixed': {
        'seed': 0,
        'n_chunks': 20,
        'eval_interval': 5,
        'eval_episodes': 50,
        'score_interval': 25,   # score more often so we see CCE active in 20 chunks
    },
}

# Claim 2 main — 5 algorithms × 10 seeds × 300 chunks.
# Submit after smoke test confirms CCE shows a learning advantage over baselines.
CLAIM2_MAIN = {
    'name': 'claim2_main',
    'env_key': 'connect_four',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                               'seed': s} for s in range(10)],
        *[{'algorithm': 'dqn',                                                       'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0, 'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',             'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',       'seed': s} for s in range(10)],
    ],
    'fixed': {'n_chunks': 38, 'exploration_fraction': 0.1},  # 38 × 10752 ≈ 400k transitions; epsilon hits 0.05 at chunk 4
}

# Claim 2 rule-based — 5 algorithms × 10 seeds × 100 chunks vs rule-based opponent.
# Win→Block→Fork opponent creates high-consequence moments that CCE is designed to exploit.
# Submit after smoke passes (opponent fires correctly, AvgR near 0 at chunk 0).
CLAIM2_RULEBASED = {
    'name': 'claim2_rulebased',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                                 'seed': s} for s in range(10)],
        *[{'algorithm': 'dqn',                                                         'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0,  'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',              'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',        'seed': s} for s in range(10)],
    ],
    'fixed': {
        'n_chunks': 100,
        'exploration_fraction': 0.1,
        'opponent': 'rule_based',
    },
}

# Claim 2 MCTS — 5 algorithms × 10 seeds × 100 chunks vs mcts_32 opponent.
# Primary Claim 2 experiment: citable MCTS baseline, same train+eval opponent.
# GPU benchmark: mcts_32 ≈ 7x rule_based per chunk (feasible on T4).
CLAIM2_MCTS = {
    'name': 'claim2_mcts',
    'env_key': 'connect_four',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                                 'seed': s} for s in range(10)],
        *[{'algorithm': 'dqn',                                                         'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0,  'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',              'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',        'seed': s} for s in range(10)],
    ],
    'fixed': {
        'n_chunks': 100,
        'exploration_fraction': 0.1,
        'opponent': 'mcts',
        'mcts_n_sims': 32,
    },
}

# Claim 2 MCTS v2 — cf_horizon=42 (full game length), cf_n_rollouts=30.
# v1 used cf_horizon=30 (too short — truncated mid-game) and cf_n_rollouts=16 (too noisy).
CLAIM2_MCTS_V2 = {
    'name': 'claim2_mcts_v2',
    'env_key': 'connect_four',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                                 'seed': s} for s in range(10)],
        *[{'algorithm': 'dqn',                                                         'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0,  'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',             'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',       'seed': s} for s in range(10)],
    ],
    'fixed': {
        'n_chunks': 100,
        'exploration_fraction': 0.1,
        'opponent': 'mcts',
        'mcts_n_sims': 32,
        'cf_horizon': 42,
        'cf_n_rollouts': 30,
    },
}

EXPERIMENTS = {
    'smoke': SMOKE,
    'claim2_main': CLAIM2_MAIN,
    'claim2_rulebased': CLAIM2_RULEBASED,
    'claim2_mcts': CLAIM2_MCTS,
    'claim2_mcts_v2': CLAIM2_MCTS_V2,
}


def generate_runs(experiment):
    fixed = experiment.get('fixed', {})
    runs = experiment.get('runs', [])
    result = []
    for run in runs:
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
    script_path = os.path.join(script_dir, 'train_connect_four.sh')

    print(f"Experiment: {experiment_name}  ({len(runs)} runs)")
    print()

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
            wait_for_slot(max_concurrent, job_ids=set(manifest.keys()))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help=f'One of: {", ".join(EXPERIMENTS.keys())}')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-concurrent', type=int, default=None,
                        help='Max jobs in squeue at once (default: no limit)')
    args = parser.parse_args()
    submit_experiment(args.experiment, dry_run=args.dry_run, max_concurrent=args.max_concurrent)
