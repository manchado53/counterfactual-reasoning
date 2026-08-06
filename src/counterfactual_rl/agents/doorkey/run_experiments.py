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

# ── Lava variants (DoorKey-6x6 with 1 lava tile) ──────────────────────────────
# The no-lava experiments above gave a capped C1 (rho 0.43) and a C2 null, both traced to
# DoorKey having no catastrophe: every wrong action was recoverable, so the oracle's
# action-value gaps stayed small everywhere. Lava (walkable but fatal) restores the
# FrozenLake-hole structure — lava-adjacent states show a ~4.2x larger oracle action-gap
# than the rest of the state space.
#
# An 8x8 4-lava layout was tried FIRST and abandoned: it is unlearnable. A random policy
# scored ZERO goals in 20k episodes (vs 0.215% on the no-lava 6x6 that trains fine), because
# the 19-step three-stage route is past what random exploration completes — and that held even
# with a single lava tile placed far from the route, so it was path length, not lava density.
# The plain-DQN sanity gate confirmed it empirically (epsilon decayed to 0.05 with win rate
# still 0.0%). Hence: catastrophe on the SHORT map. See envs/doorkey.py DOORKEY_6x6_LAVA.

# Budget note: lava roughly HALVES episode length (deaths end episodes early — measured
# ~30 env steps/episode with lava vs ~61 without). Since the epsilon schedule is denominated
# in EPISODES, the no-lava budget (15k episodes / 7.5k decay) gives the lava agent only about
# half the exploration experience — on a task that is also half as likely to stumble into the
# goal (0.110% vs 0.215% of random episodes). At the no-lava budget only 1 of 2 sanity seeds
# solved it (the other hit epsilon=0.05 still at 0% and stalled), so the budget is doubled to
# restore an exploration phase comparable to the no-lava runs.
_LAVA_BASE = {
    'layout_name': '6x6_lava',
    'n_episodes': 30000,
    'epsilon_decay_episodes': 15000,
    'vectorized': True,
    'max_episode_steps': 50,
    'cf_horizon': 60,
}

# Sanity: can plain DQN still solve it with lava punishing exploration? 4 seeds — with lava
# this is a reliability question, not a yes/no one (at the old halved budget it was 1/2).
DOORKEY_LAVA_SANITY = {
    'name': 'doorkey_lava_sanity',
    'runs': [{'algorithm': 'dqn-uniform', 'seed': s} for s in range(4)],
    'fixed': {**_LAVA_BASE, 'slip_prob': 0.0},
}

# Claim 1 (lava) — stochastic. slip=0.1 from a pre-training sweep using the OPTIMAL policy as
# the rollout policy. On a 150-state sample rho is essentially FLAT across slip 0.05-0.2
# (0.64-0.73, two seeds each), so 0.1 is a mid-range pick rather than a tuned optimum. (An
# earlier 60-state sample suggested a peak at 0.1 with decline after; that was sampling noise
# and did not survive the larger sample — recorded here so it isn't re-derived as fact.)
# slip=0.0 remains degenerate (CCE std 0.000) even WITH lava, so Claim 1 still requires
# stochastic dynamics: lava alone does not rescue the deterministic case.
DOORKEY_LAVA_CLAIM1 = {
    'name': 'doorkey_lava_claim1',
    'env_key': 'doorkey',
    'runs': [{'algorithm': 'dqn', 'seed': s} for s in range(3)],
    'fixed': {**_LAVA_BASE, 'slip_prob': 0.1, 'n_checkpoints': 100},
}

# Claim 2 (lava) — deterministic, the real test of whether catastrophe gives CCE the
# sample-efficiency edge it lacked without lava.
DOORKEY_LAVA_CLAIM2 = {
    'name': 'doorkey_lava_claim2',
    'threshold': None,
    'env_key': 'doorkey',
    'runs': [
        *[{'algorithm': 'dqn-uniform',                                               'seed': s} for s in range(10)],
        *[{'algorithm': 'dqn',                                                       'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive', 'mu': 1.0, 'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'additive',             'seed': s} for s in range(10)],
        *[{'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',       'seed': s} for s in range(10)],
    ],
    'fixed': {
        **_LAVA_BASE, 'slip_prob': 0.0,
        'mu': 0.25, 'consequence_metric': 'total_variation',
        'score_interval': 100, 'early_stop_win_rate': 0.99,
    },
}

EXPERIMENTS = {
    'doorkey_sanity': DOORKEY_SANITY,
    'doorkey_smoke': DOORKEY_SMOKE,
    'doorkey_claim1': DOORKEY_CLAIM1,
    'doorkey_claim2': DOORKEY_CLAIM2,
    'doorkey_lava_sanity': DOORKEY_LAVA_SANITY,
    'doorkey_lava_claim1': DOORKEY_LAVA_CLAIM1,
    'doorkey_lava_claim2': DOORKEY_LAVA_CLAIM2,
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
