"""Small runner hooks; validation imports no simulator or accelerator runtime."""
from copy import deepcopy
import math
from .config import DEFAULT_CONFIG

EXTRA_KEYS = {'collector_variant', 'fixed_eval_seed', 'cf_action_prob_temperature',
              'score_on_insert', 'normalise_priorities', 'early_stop_win_rate',
              'experiment', 'training_budget_kind', 'blocks', 'eval_every_blocks'}


def validate_config(config):
    unknown = set(config) - set(DEFAULT_CONFIG) - EXTRA_KEYS
    missing = set(DEFAULT_CONFIG) - set(config)
    if unknown or missing:
        raise ValueError(f'Unknown settings: {sorted(unknown)}; unresolved: {sorted(missing)}')
    if config.get('collector_variant') not in ('old', 'fixed', 'sequential'):
        raise ValueError('collector_variant must be old, fixed, or sequential')
    if config['algorithm'] not in ('dqn-uniform', 'dqn', 'consequence-dqn'):
        raise ValueError('Unsupported FrozenLake learner')
    if config['map_name'] not in ('4x4', '8x8') or type(config['vectorized']) is not bool:
        raise ValueError('Built-in map and explicit collector required')
    if config['vectorized'] != (config['collector_variant'] != 'sequential'):
        raise ValueError('Collector/vectorized mismatch')
    if not isinstance(config['is_slippery'], bool):
        raise ValueError('is_slippery must be boolean')
    p = config['slip_prob']
    if p is not None and (not isinstance(p, (int, float)) or not 0 <= p <= 1):
        raise ValueError('slip_prob must be null or in [0,1]')
    for key in ('n_envs', 'collect_steps', 'n_episodes', 'batch_size', 'buffer_capacity',
                'eval_episodes', 'cf_horizon', 'cf_n_rollouts', 'n_steps_per_update',
                'target_update_freq', 'hidden_dim', 'n_layers', 'save_every',
                'score_interval', 'n_score_sample', 'epsilon_decay_episodes'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('gamma', 'cf_gamma', 'epsilon_start', 'epsilon_end', 'mu'):
        if not isinstance(config[key], (float, int)) or not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    for key in ('alpha', 'mu_c', 'mu_delta'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Invalid {key}')
    if config['batch_size'] > config['buffer_capacity']:
        raise ValueError('Batch exceeds replay capacity')
    if config['priority_mixing'] not in ('additive', 'multiplicative'):
        raise ValueError('Unsupported priority mixing')
    if set(config['PER_parameters']) != {'eps', 'beta', 'maximum_priority'}:
        raise ValueError('Unknown or missing PER parameters')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32 - 730001:
        raise ValueError('Invalid seed')
    if config['collector_variant'] == 'fixed':
        for key in EXTRA_KEYS - {'experiment'}:
            if key not in config:
                raise ValueError(f'Unresolved setting: {key}')
        for key in ('blocks', 'eval_every_blocks'):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(config['fixed_eval_seed']) is not int or not 0 <= config['fixed_eval_seed'] < 2**32:
            raise ValueError('Fixed evaluation seed required')
        if config['training_budget_kind'] != 'env_steps' or config['early_stop_win_rate'] is not None:
            raise ValueError('Fixed transfer uses a step budget with no early stop')
        if (config['score_on_insert'] is not True or config['normalise_priorities'] != 'rank'
                or config['consequence_metric'] != 'wasserstein'
                or config['consequence_aggregation'] != 'weighted_mean'):
            raise ValueError('Unsupported fixed transfer scoring protocol')
        if not math.isfinite(config['cf_action_prob_temperature']) or config['cf_action_prob_temperature'] <= 0:
            raise ValueError('Positive finite action temperature required')
        size = config['n_envs'] * config['collect_steps']
        if not config['batch_size'] <= size <= config['buffer_capacity']:
            raise ValueError('Transfer block must fit replay and fill one batch')
    else:
        if config.get('training_budget_kind') != 'episodes':
            raise ValueError('Old collector uses an episode budget')
        unsupported = set(config) & (EXTRA_KEYS - {'collector_variant', 'training_budget_kind',
                                                  'early_stop_win_rate', 'experiment'})
        if unsupported:
            raise ValueError(f'Old collector ignores these settings: {sorted(unsupported)}')
    return config


def build_agent(config):
    validate_config(config)
    config = deepcopy(config)
    if config['collector_variant'] == 'fixed':
        from .runner import FixedCCE, FixedDQN
        return (FixedCCE if config['algorithm'] == 'consequence-dqn' else FixedDQN)(config)
    if config['collector_variant'] == 'sequential':
        from .legacy.dqn import FrozenLakeDQN
        from .legacy.consequence_dqn import FrozenLakeConsequenceDQN
        return (FrozenLakeConsequenceDQN if config['algorithm'] == 'consequence-dqn'
                else FrozenLakeDQN)(config)
    from .legacy.dqn_vectorized import FrozenLakeDQNVectorized
    from .legacy.consequence_dqn_vectorized import FrozenLakeConsequenceDQNVectorized
    return (FrozenLakeConsequenceDQNVectorized if config['algorithm'] == 'consequence-dqn'
            else FrozenLakeDQNVectorized)(config)


def smoke_config(algorithm='consequence-dqn'):
    """Tiny branch exercise, explicitly not the historical research recipe."""
    return {**deepcopy(DEFAULT_CONFIG), 'algorithm': algorithm, 'map_name': '4x4',
            'collector_variant': 'fixed', 'vectorized': True, 'fixed_eval_seed': 20261001,
            'cf_action_prob_temperature': .5, 'score_on_insert': True,
            'normalise_priorities': 'rank', 'consequence_metric': 'wasserstein',
            'early_stop_win_rate': None, 'training_budget_kind': 'env_steps',
            'experiment': 'frozenlake_migration_smoke', 'blocks': 4, 'eval_every_blocks': 2,
            'n_envs': 2, 'collect_steps': 4, 'buffer_capacity': 16, 'batch_size': 4,
            'hidden_dim': 8, 'n_layers': 1, 'cf_horizon': 4, 'cf_n_rollouts': 4,
            'eval_episodes': 4, 'target_update_freq': 8, 'n_checkpoints': 0}


def smoke_overrides(config):
    """Bound compute only; keep task, collector, reward and sampling rules."""
    overrides = {'n_envs': 2, 'collect_steps': 4, 'buffer_capacity': 16,
                 'batch_size': 4, 'hidden_dim': 8, 'n_layers': 1,
                 'cf_horizon': 4, 'cf_n_rollouts': 4, 'eval_episodes': 4,
                 'target_update_freq': 8, 'n_checkpoints': 0,
                 'n_score_sample': 4, 'score_interval': 1}
    if config['collector_variant'] == 'fixed':
        overrides.update(blocks=4, eval_every_blocks=2)
    elif config['collector_variant'] in ('old', 'sequential'):
        overrides.update(n_episodes=6, eval_interval=2, save_every=6)
    else:
        raise ValueError('Unknown collector variant')
    return overrides
