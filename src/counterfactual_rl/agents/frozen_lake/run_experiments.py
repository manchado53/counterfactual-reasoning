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


# ── Buffer reference (issue #18, Jeremy 09/03 action item 2) ──────────────────
# Measure what the FrozenLake replay buffer CONTAINS, using the instruments that
# landed 2026-09-03, so the JaxNav numbers have a measured comparison instead of an
# arithmetic one. JaxNav has been measured to death; FL never has.
#
# The arithmetic to be checked:
#     coverage = n_score_sample / (score_interval x n_steps_per_update)
#              = 128 / (100 x 4) = 32% per lifetime -> ~14.4% on the belt
#     vs JaxNav 64 / (250 x 16) = 1.6% -> 0.85% measured
#
# TWO DELIBERATE DEPARTURES from claim2_no_slip, both because this measures the
# BUFFER rather than reproducing the headline win:
#   * early_stop_win_rate disabled -- an early stop truncates exactly the
#     post-convergence stretch where the zero-rate is most informative.
#   * eval_interval tightened to 150 for a denser trajectory.
# Everything else matches the config that produced the paper's FL-det result.
#
# WHY THE ZERO-RATE MATTERS MOST. A 3k-episode CPU smoke showed FL-det at 16.7%
# coverage (formula validated) but 100% of measured scores exactly ZERO, because the
# agent had not yet reached the goal. The trained-checkpoint probe showed 0/64 dead.
# So the zero-rate is a TRAJECTORY, and JaxNav-vs-FL must be compared at matched
# competence, not matched episode count.
BUFFER_REFERENCE = {
    'name': 'buffer_reference',
    'env_key': 'frozen_lake',
    'runs': [
        *[{'algorithm': 'dqn-uniform', 'seed': s} for s in range(3)],
        *[{'algorithm': 'dqn', 'seed': s} for s in range(3)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',
           'seed': s} for s in range(3)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0,
           'seed': s} for s in range(3)],
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
        'early_stop_win_rate': None,     # see note above
        'eval_interval': 150,
        'n_buffer_snapshots': 10,
    },
}



# ── Coverage dose-response (issue #22) ───────────────────────────────────────
# THE experiment the coverage story needs. Everything so far is CORRELATIONAL across
# two environments -- FrozenLake 18.9% coverage / CCE wins vs JaxNav 0.85% / CCE null --
# and those two differ on THREE axes at once (coverage, zero-rate, rollout policy), so
# n=2 environments cannot attribute the difference to any of them.
#
# Here coverage is a DIAL in one environment that has a positive control. Only
# n_score_sample varies; score_interval, n_steps_per_update, rollouts, horizon and every
# training knob are held. Coverage is set by
#     lambda = n_score_sample / (score_interval x n_steps_per_update) = n / 400
# and the instantaneous (on-the-belt) value is calibrated against the one point we have
# actually measured, n=128 -> 18.9% (buffer_reference, 2026-09-08):
#
#     n_score_sample      16     32     64    128    256
#     coverage          2.6%   5.1%   9.9%  18.9%  34.3%     <- 13x range
#
# WHY JAXNAV CANNOT DO THIS: its paired seed sd is ~30pp, so resolving any win-rate
# effect needs ~290 seed-pairs. FrozenLake-det is bimodal (solve or don't), so the
# readout is a SOLVE RATE over seeds, which 6 seeds x 5 levels can show as a trend even
# though no single pairwise comparison would be significant.
#
# Read the TREND across levels, not any pair. n=6 per level.
# early_stop stays disabled, matching buffer_reference, so "solved" means the run ever
# reached >=95% rather than "the run was halted at 95%".
COVERAGE_DOSE = {
    'name': 'coverage_dose',
    'env_key': 'frozen_lake',
    'runs': [
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',
           'n_score_sample': n, 'seed': s}
          for n in (16, 32, 64, 128, 256) for s in range(6)],
        # uniform control, unaffected by n_score_sample (it never scores)
        *[{'algorithm': 'dqn-uniform', 'seed': s} for s in range(6)],
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
        'early_stop_win_rate': None,
        'eval_interval': 150,
        'n_buffer_snapshots': 8,
    },
}



# ── Coverage dose-response, EXTENSION to 26 seeds/level (issue #22) ──────────
# The n=6 pilot (coverage_dose, 2026-09-08) got the dose axis exactly right --
# predicted coverage 2.6/5.1/9.9/18.9/34.3%, measured 2.6/5.2/10.0/18.9/33.7% -- and a
# solve rate rising 1/6 -> 5/6 across a 13x coverage range. But:
#     trend over 30 runs   r=+0.375  p=0.041   significant
#     endpoints 1/6 vs 5/6 Fisher    p=0.080   NOT significant
#     NOT monotone: level 64 gave 1/6 while level 32 gave 3/6
# At n=6 the binomial SE on a solve rate is ~0.20, so that dip is 1.6 SE -- noise. Two
# tests on the same 30 runs disagreeing at the 0.05 line is not a defensible result.
#
# This adds seeds 6..25, giving 26 per level when pooled with the pilot. Powered to
# z>4 on the endpoint contrast, and enough to say whether the dip at level 64 is real.
# Same fixed config as the pilot so the two merge directly.
COVERAGE_DOSE_EXT = {
    'name': 'coverage_dose_ext',
    'env_key': 'frozen_lake',
    'runs': [
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',
           'n_score_sample': n, 'seed': s}
          for n in (16, 32, 64, 128, 256) for s in range(6, 26)],
        *[{'algorithm': 'dqn-uniform', 'seed': s} for s in range(6, 26)],
    ],
    'fixed': {
        'map_name': '8x8', 'is_slippery': False, 'n_episodes': 15000, 'mu': 0.25,
        'consequence_metric': 'total_variation', 'epsilon_decay_episodes': 7500,
        'score_interval': 100, 'vectorized': True, 'cf_horizon': 200,
        'early_stop_win_rate': None, 'eval_interval': 150,
        'n_buffer_snapshots': 4,     # lighter: the pilot already has the buffer detail
    },
}


EXPERIMENTS = {
    'smoke_test': SMOKE_TEST,
    'full_smoke': FULL_SMOKE,
    'pilot': PILOT,
    'claim1_dqn': CLAIM1_DQN,
    'claim2_main': CLAIM2_MAIN,
    'claim2_main_remaining': CLAIM2_MAIN_REMAINING,
    'claim2_cce_rerun': CLAIM2_CCE_RERUN,
    'claim2_cce_multiplicative': CLAIM2_CCE_MULTIPLICATIVE,
    'claim2_no_slip': CLAIM2_NO_SLIP,
    'vec_smoke': VEC_SMOKE,
    'buffer_reference': BUFFER_REFERENCE,
    'coverage_dose': COVERAGE_DOSE,
    'coverage_dose_ext': COVERAGE_DOSE_EXT,
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
    # src/ of whichever checkout this module was imported from -- main repo or worktree.
    repo_src = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    log_dir = os.path.join(script_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

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
        # Pin the job to THIS checkout. Without REPO_SRC the sbatch script falls back
        # to the main repo, so a sweep submitted from a worktree trains main-repo code
        # and silently produces runs without whatever the worktree was testing.
        # --export=ALL keeps the rest of the environment (the bare form does not).
        cmd = ['sbatch',
               f'--output={log_dir}/train_%j.out',
               f'--export=ALL,CONFIG_OVERRIDES_B64={encoded},REPO_SRC={repo_src}',
               script_path]
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
