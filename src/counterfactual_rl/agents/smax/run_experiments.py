"""Submit experiment sweeps to SLURM and save manifest files.

Usage:
    python run_experiments.py smoke_test --dry-run    # Preview jobs
    python run_experiments.py metric_sweep            # Submit all 36
    python run_experiments.py algorithm_comparison    # Submit all 27
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import date

from experiments import EXPERIMENTS, generate_runs


def submit_experiment(experiment_name, dry_run=False):
    if experiment_name not in EXPERIMENTS:
        print(f"Error: unknown experiment '{experiment_name}'")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    experiment = EXPERIMENTS[experiment_name]
    runs = generate_runs(experiment)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'train_smax_dqn.sh')

    print(f"Experiment: {experiment_name}")
    print(f"Runs: {len(runs)}")
    print()

    if dry_run:
        for i, overrides in enumerate(runs):
            overrides_json = json.dumps(overrides)
            encoded = base64.b64encode(overrides_json.encode()).decode()
            print(f"  [{i+1:3d}] CONFIG_OVERRIDES_B64={encoded}")
            print(f"         -> {overrides}")
        print(f"\n{len(runs)} jobs (dry run — nothing submitted)")
        return

    # Submit all jobs
    manifest = {}
    date_str = date.today().isoformat()          # e.g. "2026-04-12"
    month_str = date_str[:7]                     # e.g. "2026-04"
    exp_name = f"{experiment_name}_{date_str}"   # e.g. "full_algorithm_comparison_2026-04-12"
    exp_dir = os.path.join(script_dir, 'experiments', month_str, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    for i, overrides in enumerate(runs):
        overrides_json = json.dumps(overrides)
        encoded = base64.b64encode(overrides_json.encode()).decode()
        cmd = [
            'sbatch',
            f'--export=CONFIG_OVERRIDES_B64={encoded}',
            script_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  [{i+1:3d}] FAILED: {result.stderr.strip()}")
            continue

        # Parse job ID from "Submitted batch job 123456"
        job_id = result.stdout.strip().split()[-1]
        manifest[job_id] = overrides
        print(f"  [{i+1:3d}] Job {job_id}: {overrides}")

    # Save manifest
    manifest_path = os.path.join(exp_dir, f"{exp_name}.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(manifest)}/{len(runs)} jobs submitted")
    print(f"Manifest saved to {manifest_path}")

    # Submit summary job that runs after all training jobs complete
    if manifest:
        job_ids = list(manifest.keys())
        dependency = ':'.join(job_ids)
        summarize_path = os.path.join(script_dir, 'summarize_experiment.py')
        python_bin = os.path.expanduser('~/.conda/envs/counterfactual/bin/python')
        summary_cmd = [
            'sbatch',
            '--job-name=summarize',
            f'--output={os.path.join(script_dir, "logs", "summarize_%j.out")}',
            '--partition=teaching',
            '--nodes=1',
            '--cpus-per-task=1',
            '--mem=4G',
            '--time=00:10:00',
            f'--dependency=afterany:{dependency}',
            '--wrap',
            f'{python_bin} {summarize_path} {manifest_path}',
        ]
        result = subprocess.run(summary_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            summary_job_id = result.stdout.strip().split()[-1]
            print(f"Summary job {summary_job_id} queued (runs after all training jobs)")
        else:
            print(f"Warning: failed to submit summary job: {result.stderr.strip()}")
            print(f"Run manually: python summarize_experiment.py {manifest_path}")

        # Submit timing comparison job (runs after all training jobs)
        timing_cmd = [
            'sbatch',
            '--job-name=timing-compare',
            f'--output={os.path.join(script_dir, "logs", "timing_%j.out")}',
            '--partition=teaching',
            '--nodes=1',
            '--cpus-per-task=1',
            '--mem=4G',
            '--time=00:10:00',
            f'--dependency=afterany:{dependency}',
            '--wrap',
            f'{python_bin} -m counterfactual_rl.training.smax.shared.timing.plot --manifest {manifest_path}',
        ]
        result = subprocess.run(timing_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            timing_job_id = result.stdout.strip().split()[-1]
            print(f"Timing comparison job {timing_job_id} queued (runs after all training jobs)")
        else:
            print(f"Warning: failed to submit timing job: {result.stderr.strip()}")
            print(f"Run manually: python -m counterfactual_rl.training.smax.shared.timing.plot --manifest {manifest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit experiment sweeps to SLURM')
    parser.add_argument('experiment', help=f'Experiment name: {", ".join(EXPERIMENTS.keys())}')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without submitting')
    args = parser.parse_args()

    submit_experiment(args.experiment, dry_run=args.dry_run)
