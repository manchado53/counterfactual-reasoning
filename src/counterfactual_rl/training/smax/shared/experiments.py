"""Experiment definitions for SMAX DQN sweeps."""

from itertools import product


def generate_runs(experiment):
    """Expand sweep dict into flat list of config override dicts."""
    sweep = experiment.get('sweep', {})
    fixed = experiment.get('fixed', {})
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    runs = []
    for combo in product(*values):
        overrides = dict(zip(keys, combo))
        overrides.update(fixed)
        runs.append(overrides)
    return runs


# --- Experiments ---

# 1. Metric comparison — find best divergence metric (36 runs)
METRIC_SWEEP = {
    'name': 'metric_sweep',
    'sweep': {
        'consequence_metric': ['kl_divergence', 'jensen_shannon',
                               'total_variation', 'wasserstein'],
        'scenario': ['3m', '5m_vs_6m', '2s3z'],
        'seed': [0, 1, 2],
    },
    'fixed': {'algorithm': 'consequence-dqn', 'mu': 0.5,
              'n_episodes': 30000, 'epsilon_decay_episodes': 10000, 'score_interval': 100},
}

# 2. Mu sensitivity — find best mixing weight (36 runs)
#    TODO: Update 'consequence_metric' to best metric from step 1
MU_SWEEP = {
    'name': 'mu_sweep',
    'sweep': {
        'mu': [0.25, 0.5, 0.75, 1.0],
        'scenario': ['3m'],  # '5m_vs_6m', '2s3z'],
        'seed': [0, 1, 2],
    },
    'fixed': {'algorithm': 'consequence-dqn', 'consequence_metric': 'kl_divergence',
              'n_episodes': 25000, 'epsilon_decay_episodes': 10000, 'score_interval': 100},
}

# 3. Algorithm comparison — head-to-head with best config (9 runs)
#    Best metric: kl_divergence (step 1), best mu: 0.25 (step 2)
#    Focused on 3m — only scenario that learns with current hyperparams
ALGORITHM_COMPARISON = {
    'name': 'algorithm_comparison',
    'sweep': {
        'algorithm': ['dqn-uniform', 'dqn', 'consequence-dqn'],
        'scenario': ['3m'],
        'seed': [0, 1, 2],
    },
    'fixed': {'mu': 0.25, 'consequence_metric': 'kl_divergence',
              'n_episodes': 25000, 'epsilon_decay_episodes': 10000, 'score_interval': 100},
}

# 4. Quick algorithm comparison — all 3 algos, 1 seed, 500 episodes (3 runs)
ALGORITHM_COMPARISON_QUICK = {
    'name': 'algorithm_comparison_quick',
    'sweep': {
        'algorithm': ['dqn-uniform', 'dqn', 'consequence-dqn'],
    },
    'fixed': {'scenario': '3m', 'seed': 0, 'mu': 0.5,
              'consequence_metric': 'kl_divergence',
              'n_episodes': 10000, 'epsilon_decay_episodes': 200,
              'eval_interval': 100, 'score_interval': 50,
              'diagnostics_enabled': True},
}

# 5. Quick pipeline test (2 runs)
SMOKE_TEST = {
    'name': 'smoke_test',
    'sweep': {'seed': [0, 1]},
    'fixed': {'algorithm': 'dqn-uniform', 'scenario': '3m',
              'n_episodes': 200, 'eval_interval': 50},
}

# 6. Algorithm comparison on 3s5z — uses scenario preset for architecture (9 runs)
ALGORITHM_COMPARISON_3S5Z = {
    'name': 'algorithm_comparison_3s5z',
    'sweep': {
        'algorithm': ['dqn-uniform', 'dqn', 'consequence-dqn'],
        'seed': [0],
    },
    'fixed': {'scenario': '3s5z', 'mu': 0.5, 'consequence_metric': 'kl_divergence',
              'n_episodes': 30000, 'epsilon_decay_episodes': 10000, 'score_interval': 300},
}

# 7. Mixing comparison — additive (Eq 4) vs multiplicative (Eq 5), 3 seeds (6 runs)
MIXING_COMPARISON = {
    'name': 'mixing_comparison',
    'sweep': {
        'priority_mixing': ['additive', 'multiplicative'],
        'seed': [0, 1, 2],
    },
    'fixed': {'algorithm': 'consequence-dqn', 'scenario': '3m',
              'mu': 0.25, 'mu_c': 1.0, 'mu_delta': 1.0,
              'consequence_metric': 'wasserstein',
              'n_episodes': 25000, 'epsilon_decay_episodes': 10000, 'score_interval': 100},
}

EXPERIMENTS = {
    'metric_sweep': METRIC_SWEEP,
    'mu_sweep': MU_SWEEP,
    'algorithm_comparison': ALGORITHM_COMPARISON,
    'algorithm_comparison_quick': ALGORITHM_COMPARISON_QUICK,
    'algorithm_comparison_3s5z': ALGORITHM_COMPARISON_3S5Z,
    'smoke_test': SMOKE_TEST,
    'mixing_comparison': MIXING_COMPARISON,
}
# Total: 36 + 12 + 9 = 57 runs (+ 2 smoke test + 6 mixing comparison)
#
# Run order:
#   1. metric_sweep → find best metric           [DONE: kl_divergence]
#   2. mu_sweep → find best mu                    [DONE: mu=0.25]
#   3. algorithm_comparison → final comparison    [3m only, 9 runs]
#   4. mixing_comparison → additive vs multiplicative [6 runs]
