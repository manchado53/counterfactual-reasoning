"""Submit FrozenLake experiment sweeps to SLURM and save manifest files.

Usage:
    python run_experiments.py pilot --dry-run
    python run_experiments.py pilot
    python run_experiments.py claim2_main
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import date
from itertools import product


# ── Experiment definitions ────────────────────────────────────────────────────

def _sweep(sweep_dict, fixed=None):
    """Cartesian product of sweep values merged with fixed params."""
    keys = list(sweep_dict.keys())
    combos = list(product(*[sweep_dict[k] for k in keys]))
    runs = [dict(zip(keys, c)) for c in combos]
    if fixed:
        for r in runs:
            r.update(fixed)
    return runs


# Phase 1: pilot — establishes FrozenLake 8×8 slippery threshold
PILOT = {
    'name': 'pilot',
    'runs': [{'algorithm': 'dqn-uniform', 'seed': 0, 'n_episodes': 30000, 'map_name': '8x8'}],
}

# Claim 2 main — 5 algorithms × 10 seeds
# UPDATE mu and consequence_metric after SMAX sweeps complete.
# UPDATE threshold after pilot completes.
CLAIM2_MAIN = {
    'name': 'claim2_main',
    'threshold': None,   # set after pilot
    'env_key': 'frozen_lake',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                              'seed': s} for s in range(10)],
        *[{'algorithm': 'dqn',                                                      'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0, 'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',            'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',      'seed': s} for s in range(10)],
    ],
    'fixed': {
        'map_name': '8x8',
        'n_episodes': 30000,                  # adjust after pilot
        'mu': 0.25,                           # UPDATE after SMAX mu sweep
        'consequence_metric': 'wasserstein',  # UPDATE after SMAX metric sweep
        'epsilon_decay_episodes': 10000,
        'score_interval': 300,
    },
}

SMOKE_TEST = {
    'name': 'smoke_test',
    'runs': [{'algorithm': 'dqn-uniform', 'seed': 0, 'map_name': '8x8',
              'n_episodes': 500, 'eval_interval': 50, 'eval_episodes': 20}],
}

# Full algorithm smoke test — all 5 algorithms, 1 seed, small buffer to force scoring
FULL_SMOKE = {
    'name': 'full_smoke',
    'runs': [
        {'algorithm': 'dqn-uniform'},
        {'algorithm': 'dqn'},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 0.25},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative'},
    ],
    'fixed': {
        'map_name': '8x8', 'seed': 0,
        'n_episodes': 500, 'eval_interval': 50, 'eval_episodes': 20,
        'buffer_capacity': 2000, 'score_interval': 50,
        'consequence_metric': 'wasserstein', 'mu': 0.25,
    },
}

EXPERIMENTS = {
    'smoke_test': SMOKE_TEST,
    'full_smoke': FULL_SMOKE,
    'pilot': PILOT,
    'claim2_main': CLAIM2_MAIN,
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

def submit_experiment(experiment_name, dry_run=False):
    if experiment_name not in EXPERIMENTS:
        print(f"Error: unknown experiment '{experiment_name}'")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    experiment = EXPERIMENTS[experiment_name]
    runs = generate_runs(experiment)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'train_frozen_lake_dqn.sh')

    print(f"Experiment: {experiment_name}  ({len(runs)} runs)")
    print()

    if dry_run:
        for i, overrides in enumerate(runs):
            print(f"  [{i+1:3d}] {overrides}")
        print(f"\n{len(runs)} jobs (dry run — nothing submitted)")
        threshold = experiment.get('threshold')
        if threshold is not None:
            print(f"\nAnalysis job would fire after all training:")
            print(f"  env={experiment['env_key']}  threshold={threshold}")
        return

    manifest = {}
    date_str = date.today().isoformat()
    month_str = date_str[:7]
    exp_name = f"{experiment_name}_{date_str}"
    exp_dir = os.path.join(script_dir, 'experiments', month_str, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    for i, overrides in enumerate(runs):
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

    # Submit Claim 2 analysis job if this experiment has a registered threshold
    threshold = experiment.get('threshold')
    env_key = experiment.get('env_key')
    if threshold is not None and env_key is not None and manifest:
        job_ids_str = ':'.join(manifest.keys())
        repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
        analysis_script = os.path.join(
            repo_root, 'src', 'counterfactual_rl', 'analysis', 'claim2', 'run_analysis.sh'
        )
        out_dir = os.path.join(repo_root, 'docs', 'figures', experiment_name)
        os.makedirs(out_dir, exist_ok=True)
        analysis_cmd = [
            'sbatch',
            f'--dependency=afterany:{job_ids_str}',
            (f'--export=ANALYSIS_MANIFEST={manifest_path},'
             f'ANALYSIS_ENV={env_key},'
             f'ANALYSIS_THRESHOLD={threshold},'
             f'ANALYSIS_OUT={out_dir}'),
            analysis_script,
        ]
        result = subprocess.run(analysis_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            analysis_job_id = result.stdout.strip().split()[-1]
            print(f"Analysis job {analysis_job_id} queued "
                  f"(env={env_key}, threshold={threshold})")
        else:
            print(f"Warning: failed to submit analysis job: {result.stderr.strip()}")
            print(f"Run manually: python -m counterfactual_rl.analysis.claim2.run_analysis "
                  f"--manifest {manifest_path} --env {env_key} "
                  f"--threshold {threshold} --out {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help=f'One of: {", ".join(EXPERIMENTS.keys())}')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    submit_experiment(args.experiment, dry_run=args.dry_run)
