"""Historical protocol. Validation imports no simulator."""
from copy import deepcopy
from pathlib import Path
import math
from .config import DEFAULT_CONFIG as DEFAULTS


def validate_config(config):
    expected = set(DEFAULTS) | {'collector_variant', 'training_budget_kind'}
    if set(config) != expected:
        raise ValueError(f'Unknown: {sorted(set(config)-expected)}; missing: {sorted(expected-set(config))}')
    if config['collector_variant'] != 'historical' or config['training_budget_kind'] != 'episodes':
        raise ValueError('Only the historical protocol is ported')
    if config['algorithm'] not in ('dqn-uniform', 'dqn', 'consequence-dqn'):
        raise ValueError('Unsupported algorithm')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    for key in ['batch_size', 'buffer_capacity', 'hidden_dim', 'cf_horizon', 'cf_n_rollouts', 'n_score_sample', 'score_interval', 'eval_episodes', 'save_every', 'n_episodes', 'target_update_freq', 'n_steps_per_update', 'n_layers']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['batch_size'] > config['buffer_capacity']:
        raise ValueError('Batch exceeds replay capacity')
    for key in ('gamma', 'cf_gamma', 'epsilon_start', 'epsilon_end', 'mu'):
        if not isinstance(config[key], (int, float)) or not 0 <= config[key] <= 1:
            raise ValueError(f'Invalid {key}')
    for key in ('alpha', 'mu_c', 'mu_delta'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Invalid {key}')
    if config['priority_mixing'] not in ('additive', 'multiplicative'):
        raise ValueError('Unsupported priority mixing')
    if config['consequence_metric'] not in ('total_variation', 'wasserstein', 'kl_divergence', 'jensen_shannon'):
        raise ValueError('Unsupported metric')
    if config['consequence_aggregation'] not in ('mean', 'weighted_mean', 'max'):
        raise ValueError('Unsupported aggregation')
    if set(config['PER_parameters']) != {'eps', 'beta', 'maximum_priority'}:
        raise ValueError('Unresolved PER settings')
    if config['vectorized'] is not False or config['instance'] not in ('default', 'small'):
        raise ValueError('Only historical tabular CVRP/TSP is ported')
    if config['capacity'] is not None and (type(config['capacity']) is not int or config['capacity'] < 1):
        raise ValueError('Invalid capacity')
    if not math.isfinite(config['travel_noise']) or config['travel_noise'] < 0:
        raise ValueError('Invalid travel_noise')
    return config


def smoke_overrides(config):
    validate_config(config)
    return {'batch_size': 2, 'buffer_capacity': 16, 'hidden_dim': 8, 'cf_horizon': 2, 'cf_n_rollouts': 2, 'n_score_sample': 2, 'score_interval': 1, 'eval_episodes': 1, 'eval_interval': 1, 'n_checkpoints': 0, 'save_every': 2, 'n_episodes': 2, 'target_update_freq': 2, 'n_steps_per_update': 1, 'n_layers': 1}


def evaluation_axis(config):
    # Chess logs the 1-based collection chunk in the old episode column.
    return 'episode'


def simulator_sources(config):
    from training.envs import cvrp
    result = {'cvrp.py': Path(cvrp.__file__)}
    pass
    return result


def build_agent(config):
    validate_config(config)
    config = deepcopy(config)
    from .dqn import CVRPDQN
    from .consequence_dqn import CVRPConsequenceDQN
    return (CVRPConsequenceDQN if config['algorithm'] == 'consequence-dqn' else CVRPDQN)(config)
