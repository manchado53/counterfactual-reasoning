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

from counterfactual_rl.agents.shared.slurm_throttle import wait_for_slot


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

# Claim 2 main — remaining 43 jobs (dqn-uniform seeds 0-6 already submitted)
CLAIM2_MAIN_REMAINING = {
    'name': 'claim2_main',  # same name so manifests merge into same experiment dir
    'threshold': None,
    'env_key': 'frozen_lake',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                               'seed': s} for s in range(7, 10)],
        *[{'algorithm': 'dqn',                                                       'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0, 'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',             'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',       'seed': s} for s in range(10)],
    ],
    'fixed': {
        'map_name': '8x8',
        'n_episodes': 30000,
        'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 10000,
        'score_interval': 300,
        'early_stop_win_rate': 0.97,
    },
}

# Claim 2 main — 5 algorithms × 10 seeds
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
        'n_episodes': 15000,
        'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 7500,
        'score_interval': 100,
        'vectorized': True,
        'cf_horizon': 200,
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
        'consequence_metric': 'total_variation', 'mu': 0.25,
    },
}

# Claim 1 — DQN+PER checkpointed runs for oracle correlation analysis
CLAIM1_DQN = {
    'name': 'claim1_dqn',
    'runs': [
        {'algorithm': 'dqn', 'seed': 0},
        {'algorithm': 'dqn', 'seed': 1},
        {'algorithm': 'dqn', 'seed': 2},
    ],
    'fixed': {
        'map_name': '8x8',
        'n_episodes': 15000,
        'early_stop_win_rate': 0.99,
    },
}

# Vectorized speed test — all 3 algorithm families, 1 seed, short run to benchmark wall-clock
VEC_SMOKE = {
    'name': 'vec_smoke',
    'runs': [
        {'algorithm': 'dqn-uniform'},
        {'algorithm': 'dqn'},
        {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 0.25},
    ],
    'fixed': {
        'map_name': '8x8', 'seed': 0,
        'n_episodes': 5000, 'eval_interval': 500, 'eval_episodes': 100,
        'epsilon_decay_episodes': 3000, 'score_interval': 100,
        'consequence_metric': 'total_variation', 'mu': 0.25,
        'vectorized': True, 'n_envs': 256, 'collect_steps': 128,
    },
}

# Claim 2 CCE-only resubmit — 30 CCE jobs with cf_horizon=200
# dqn-uniform + dqn already done (jobs 255448-255467); only CCE needs rerunning
CLAIM2_CCE_RERUN = {
    'name': 'claim2_main',  # same name → unified manifest
    'runs': [
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0, 'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',             'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',       'seed': s} for s in range(10)],
    ],
    'fixed': {
        'map_name': '8x8',
        'n_episodes': 15000,
        'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 7500,
        'score_interval': 100,
        'vectorized': True,
        'cf_horizon': 200,
    },
}

# CCE-multiplicative only — additive mu=1.0 + mu=0.25 already done (jobs 255495-255528)
CLAIM2_CCE_MULTIPLICATIVE = {
    'name': 'claim2_main',  # same name → unified manifest
    'runs': [
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative', 'seed': s} for s in range(10)],
    ],
    'fixed': {
        'map_name': '8x8',
        'n_episodes': 15000,
        'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 7500,
        'score_interval': 100,
        'vectorized': True,
        'cf_horizon': 200,
    },
}

# Claim 2 non-slippery — identical to claim2_main but is_slippery=False
# Kept separate so it doesn't pollute the slippery manifest.
CLAIM2_NO_SLIP = {
    'name': 'claim2_no_slip',
    'env_key': 'frozen_lake',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                              'seed': s} for s in range(25)],
        *[{'algorithm': 'dqn',                                                      'seed': s} for s in range(25)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0, 'seed': s} for s in range(25)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',            'seed': s} for s in range(25)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',      'seed': s} for s in range(25)],
    ],
    'fixed': {
        'map_name': '8x8',
        'is_slippery': False,
        'n_episodes': 15000,
        'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 7500,
        'score_interval': 100,
        'vectorized': True,
        'cf_horizon': 200,
        'early_stop_win_rate': 0.95,
    },
}

# Claim 2 graded stochasticity — sweep slip_prob to trace the CCE-vs-PER advantage
# against environment noise (Theorem 3: the advantage should grow as slip falls).
# Endpoints reproduce known paper results as a built-in sanity check:
#   slip_prob=0.0    == claim2_no_slip (deterministic, CCE-mul wins big)
#   slip_prob=0.666  == claim2_main    (full slip, null)
# Additive-mixing cell dropped per 07/30 meeting -> 4 algos x 10 seeds x 5 levels = 200 runs.
# `fixed` block matches the paper's FrozenLake runs exactly (see paper/repro/manifests).
_GRADED_SLIP_LEVELS = [0.0, 0.166, 0.333, 0.5, 0.666]
_GRADED_SLIP_ALGOS = [
    {'algorithm': 'dqn-uniform'},                                              # uniform replay
    {'algorithm': 'dqn'},                                                      # PER
    {'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0},  # CCE-only
    {'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative'},       # CCE+TD (mul)
]
CLAIM2_GRADED_SLIP = {
    'name': 'claim2_graded_slip',
    'env_key': 'frozen_lake',
    'runs': [
        {**algo, 'slip_prob': p, 'seed': s}
        for p in _GRADED_SLIP_LEVELS
        for algo in _GRADED_SLIP_ALGOS
        for s in range(10)
    ],
    'fixed': {
        'map_name': '8x8',
        'n_episodes': 15000,
        'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 7500,
        'score_interval': 100,
        'vectorized': True,
        'cf_horizon': 200,
        'early_stop_win_rate': 0.95,
    },
}

# Dense follow-up to claim2_graded_slip (2026-08-03). That sweep put 5 points on the
# noise axis and only ONE of them moved: CCE-mul beat PER at slip=0 (P=0.65) and tied
# everywhere else. With one moving point you cannot tell a steep decay from a knife
# edge at exactly determinism — and those are different papers. So: sample densely
# where the arms still separate.
#
# Where the signal is (from the 2026-08-03 clean numbers, dead seeds dropped):
#     slip 0.000   uniform 0.33  CCE-only 0.80  PER 0.67  mul 1.00   spread 0.67
#     slip 0.166   uniform 0.15  CCE-only 0.48  PER 0.99  mul 0.98   spread 0.84
#     slip 0.333   uniform 0.97  CCE-only 0.96  PER 0.97  mul 0.96   spread 0.01  <- ceiling
#     slip 0.500   uniform 0.92  CCE-only 0.96  PER 0.93  mul 0.88   spread 0.08  <- ceiling
#     slip 0.666   uniform 0.66  CCE-only 0.73  PER 0.66  mul 0.71   spread 0.07  <- floor
# Slip aids exploration, so by 0.333 every arm hits ~0.96 and no gap can appear.
# The measurable window is 0.0 - 0.25; that is where the extra levels go.
_GRADED_SLIP_DENSE_MAIN = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.133, 0.166,
                           0.20, 0.25, 0.333, 0.5, 0.666]

# Falsification probe ABOVE the noise peak. Outcome probs are [p/2, 1-p, p/2], so
# entropy peaks at p=2/3 ([1/3,1/3,1/3]) and FALLS after it: p=0.9 -> [.45,.1,.45]
# (H=0.95) and p=1.0 -> [.5,0,.5] (H=0.69). Past the peak, slip rises while noise
# drops, which is the one place "CCE needs LOW NOISE" and "CCE needs DETERMINISM"
# predict opposite things. Caveat: past 2/3 the intended action becomes the least
# likely outcome (anti-informative controls), so if every arm collapses here the
# probe is uninformative rather than evidence either way.
_GRADED_SLIP_DENSE_PROBE = [0.8, 0.9, 1.0]

_GRADED_SLIP_DENSE_LEVELS = _GRADED_SLIP_DENSE_MAIN + _GRADED_SLIP_DENSE_PROBE

# 20 seeds, not 10: outcomes here are close to all-or-nothing per seed, and at 10
# seeds the best CI on the headline metric was P=0.65 [0.55, 0.70] — too wide to
# resolve a slope across neighbouring levels.
CLAIM2_GRADED_SLIP_DENSE = {
    'name': 'claim2_graded_slip_dense',
    'env_key': 'frozen_lake',
    'runs': [
        {**algo, 'slip_prob': p, 'seed': s}
        for p in _GRADED_SLIP_DENSE_LEVELS
        for algo in _GRADED_SLIP_ALGOS
        for s in range(20)
    ],
    # Identical to CLAIM2_GRADED_SLIP so the shared levels stay comparable and the
    # endpoints still reproduce the paper's no-slip / full-slip caches.
    'fixed': {
        'map_name': '8x8',
        'n_episodes': 15000,
        'mu': 0.25,
        'consequence_metric': 'total_variation',
        'epsilon_decay_episodes': 7500,
        'score_interval': 100,
        'vectorized': True,
        'cf_horizon': 200,
        'early_stop_win_rate': 0.95,
    },
}

EXPERIMENTS = {
    'claim2_graded_slip_dense': CLAIM2_GRADED_SLIP_DENSE,
    'smoke_test': SMOKE_TEST,
    'full_smoke': FULL_SMOKE,
    'pilot': PILOT,
    'claim1_dqn': CLAIM1_DQN,
    'claim2_main': CLAIM2_MAIN,
    'claim2_main_remaining': CLAIM2_MAIN_REMAINING,
    'claim2_cce_rerun': CLAIM2_CCE_RERUN,
    'claim2_cce_multiplicative': CLAIM2_CCE_MULTIPLICATIVE,
    'claim2_no_slip': CLAIM2_NO_SLIP,
    'claim2_graded_slip': CLAIM2_GRADED_SLIP,
    'vec_smoke': VEC_SMOKE,
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
    parser.add_argument('--max-concurrent', type=int, default=None,
                        help='Max jobs in squeue at once (default: no limit)')
    args = parser.parse_args()
    submit_experiment(args.experiment, dry_run=args.dry_run, max_concurrent=args.max_concurrent)
