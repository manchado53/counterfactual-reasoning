import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ARMS = ('uniform', 'per', 'cce_mul', 'cce_add', 'cce_only')
DECAY_TRANSFER_ARMS = ('uniform', 'per', 'cce_decay')
PROCEDURAL_BENCHMARK_ARMS = ('uniform', 'per', 'cce_decay')

# Every field here is copied verbatim from task 33 of the JaxNav ranked-TD
# decay recipe: fixed collector, CCE-only, decay enabled, seed 0. Tests compare
# the transfer recipe back to that source row so this list cannot drift quietly.
JAXNAV_PARITY_FIELDS = frozenset({
    'gamma', 'epsilon_start', 'epsilon_end', 'epsilon_decay_episodes',
    'alpha', 'batch_size', 'buffer_capacity', 'target_update_freq',
    'n_steps_per_update', 'double_dqn', 'hidden_dim', 'n_layers',
    'n_episodes', 'eval_interval', 'eval_episodes', 'fixed_eval_seed',
    'save_every', 'n_checkpoints', 'n_buffer_snapshots', 'qdiag_enabled',
    'qdiag_every', 'PER_parameters', 'early_stop_win_rate',
    'early_stop_patience', 'early_stop_min_delta',
    'early_stop_smooth_window', 'early_stop_min_episodes', 'mu',
    'priority_mixing', 'mu_c', 'mu_delta', 'score_interval',
    'n_score_sample', 'consequence_metric', 'consequence_aggregation',
    'cf_horizon', 'cf_n_rollouts', 'cf_gamma', 'cf_rollout_temperature',
    'cf_bootstrap', 'vectorized', 'n_envs', 'collect_steps',
    'score_on_insert', 'cf_score_chunk', 'normalise_priorities',
    'cf_action_prob_temperature', 'cf_drift_probe_n',
    'cf_drift_probe_start_ep', 'decay_start', 'decay_end',
})

JAXNAV_WINNER = dict(
    gamma=.99,
    epsilon_start=1,
    epsilon_end=.05,
    epsilon_decay_episodes=62500,
    alpha=.0003,
    batch_size=128,
    buffer_capacity=100000,
    target_update_freq=2000,
    n_steps_per_update=16,
    double_dqn=True,
    hidden_dim=128,
    n_layers=3,
    n_episodes=200000,
    eval_interval=250,
    eval_episodes=1000,
    fixed_eval_seed=20260923,
    save_every=1000,
    n_checkpoints=100,
    n_buffer_snapshots=8,
    qdiag_enabled=True,
    qdiag_every=10,
    PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1),
    early_stop_win_rate=None,
    early_stop_patience=100000,
    early_stop_min_delta=.02,
    early_stop_smooth_window=5,
    early_stop_min_episodes=None,
    mu=1,
    priority_mixing='additive',
    mu_c=1,
    mu_delta=1,
    score_interval=100,
    n_score_sample=128,
    consequence_metric='wasserstein',
    consequence_aggregation='weighted_mean',
    cf_horizon=60,
    cf_n_rollouts=1,
    cf_gamma=.99,
    cf_rollout_temperature=0,
    cf_bootstrap=True,
    vectorized=True,
    n_envs=256,
    collect_steps=32,
    score_on_insert=True,
    cf_score_chunk=1280,
    normalise_priorities='rank',
    cf_action_prob_temperature=.5,
    cf_drift_probe_n=256,
    cf_drift_probe_start_ep=20000,
    decay_start=100000,
    decay_end=150000,
)


def procedural_benchmark_config(arm, seed):
    """Matched 12x12 procedural DoorKey benchmark built from the JaxNav winner."""
    if arm not in PROCEDURAL_BENCHMARK_ARMS:
        raise ValueError(f'Unsupported procedural DoorKey arm: {arm}')
    if type(seed) is not int or seed not in range(5):
        raise ValueError(f'Unsupported procedural DoorKey seed: {seed}')
    cce = arm == 'cce_decay'
    algorithm = 'consequence-dqn' if cce else ('dqn-uniform' if arm == 'uniform' else 'dqn')
    return dict(
        JAXNAV_WINNER,
        seed=seed,
        arm=arm,
        algorithm=algorithm,
        cce_enabled=cce,
        decay_enabled=cce,
        optimizer='adam',
        gradient_clip=10.,
        loss='importance-weighted-squared-td',
        importance_weights='1/(N*p), no batch rescale',
        target_copy_phase='after attempts 0,125,250,...',
        environment='minigrid-doorkey-procedural',
        protocol='procedural-doorkey-12x12',
        collector_variant='carry-safe',
        training_budget_kind='env_steps',
        size=12,
        max_episode_steps=1440,
        eval_max_steps=1440,
        observation='symbolic-7x7-partial',
        reward='minigrid-time-scaled-goal',
        total_env_steps=1_000_000,
        eval_interval_steps=25_000,
        epsilon_decay_steps=312_500,
        decay_start=500_000,
        decay_end=750_000,
        timeout_bootstrap=True,
        experiment='doorkey_procedural_12x12_20261013',
    )


def decay_transfer_config(arm, seed, layout_name):
    if arm not in DECAY_TRANSFER_ARMS:
        raise ValueError(f'Unsupported decay-transfer arm: {arm}')
    if type(seed) is not int or seed not in range(5):
        raise ValueError(f'Unsupported decay-transfer seed: {seed}')
    if layout_name not in ('6x6', '6x6_lava'):
        raise ValueError(f'Unsupported decay-transfer layout: {layout_name}')
    cce = arm == 'cce_decay'
    algorithm = 'consequence-dqn' if cce else ('dqn-uniform' if arm == 'uniform' else 'dqn')
    return dict(
        JAXNAV_WINNER,
        seed=seed,
        arm=arm,
        algorithm=algorithm,
        cce_enabled=cce,
        decay_enabled=cce,
        optimizer='adam',
        gradient_clip=10.,
        loss='importance-weighted-squared-td',
        target_copy_phase='after attempts 0,125,250,...',
        importance_weights='1/(N*p), no batch rescale',
        refresh_enabled=False,
        drift_probe_enabled=False,
        environment='custom-full-state-doorkey',
        layout_name=layout_name,
        slip_prob=0.,
        random_start='fixed',
        max_episode_steps=50,
        eval_max_steps=50,
        explore_hold_steps=1,
        observation='one-hot-state-id',
        timeout_bootstrap=True,
        protocol='jaxnav-ranked-td-decay-transfer',
        collector_variant='fixed',
        training_budget_kind='episodes',
        experiment='doorkey_jaxnav_decay_transfer_20261013',
    )

from .source_recipe import RECIPE

def config_for(arm, seed):
    row = next(r for r in RECIPE['rows'] if r['arm'] == arm and r['seed'] == seed)
    expected = dict(gamma=.99, alpha=.0003, hidden_dim=128, n_layers=3,
        batch_size=128, buffer_capacity=100000, target_update_freq=2000,
        n_steps_per_update=16, double_dqn=True, n_envs=256, collect_steps=32,
        epsilon_start=1., epsilon_end=.05, epsilon_decay_episodes=62500,
        fixed_eval_seed=20260923,
        PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1.))
    for name,value in expected.items():
        assert row['config'][name] == value, ('source recipe drift', name)
    cfg = dict(expected, seed=seed, arm=arm, cce_enabled=arm.startswith('cce'),
               optimizer='adam', gradient_clip=10., loss='importance-weighted-squared-td',
               target_copy_phase='after attempts 0,125,250,...',
               importance_weights='1/(N*p), no batch rescale')
    if cfg['cce_enabled']:
        cce = dict(consequence_metric='wasserstein', consequence_aggregation='weighted_mean',
            cf_horizon=60, cf_n_rollouts=1, cf_gamma=.99, cf_rollout_temperature=0,
            cf_bootstrap=True, score_on_insert=True, cf_score_chunk=1280,
            normalise_priorities='rank', cf_action_prob_temperature=.5,
            mu=1. if arm == 'cce_only' else .5, mu_c=1., mu_delta=1.,
            priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive')
        for name,value in cce.items():
            assert row['config'][name] == value, ('source CCE drift', name)
        cfg.update(cce, refresh_enabled=False, drift_probe_enabled=False)
    cfg.update(environment='custom-full-state-doorkey', layout_name='6x6', slip_prob=0.,
               random_start='fixed', max_episode_steps=50, eval_max_steps=50,
               explore_hold_steps=1, eval_episodes=1000, observation='one-hot-state-id',
               timeout_bootstrap=True, blocks=2000, eval_every_blocks=50)
    # All historical/inert keys remain in source_recipe.json, not runtime config.
    return cfg
