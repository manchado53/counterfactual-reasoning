"""Separate pinned budget and Option-A protocols. No simulator import for validation."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import math

VARIANTS = ('budget', 'option_a')


def validate_config(config):
    variant = config.get('learner_variant')
    if variant not in VARIANTS:
        raise ValueError('learner_variant must be budget or option_a')
    defaults = import_module(f'{__package__}.{variant}.config').DEFAULT_CONFIG
    expected = set(defaults) | {'learner_variant', 'collector_variant', 'training_budget_kind'}
    if set(config) != expected:
        raise ValueError(f'Unknown: {sorted(set(config)-expected)}; missing: {sorted(expected-set(config))}')
    if config['collector_variant'] != 'historical' or config['training_budget_kind'] != 'episodes':
        raise ValueError('Historical episode protocol required')
    if config['vectorized'] is not False:
        raise ValueError('Historical budget learners are scalar collectors')
    if config['budget_mult'] is None and config['budget_units'] is None:
        raise ValueError('Budget mode must be explicit; plain CVRP is a different port')
    if config['budget_mult'] is not None and (not math.isfinite(config['budget_mult']) or config['budget_mult'] <= 0):
        raise ValueError('budget_mult must be positive and finite')
    if config['budget_units'] is not None and (type(config['budget_units']) is not int or config['budget_units'] < 1):
        raise ValueError('budget_units must be a positive integer')
    if config['algorithm'] not in ('dqn-uniform', 'dqn', 'consequence-dqn'):
        raise ValueError('Unsupported algorithm')
    for key in ('n_episodes', 'batch_size', 'buffer_capacity', 'hidden_dim', 'n_layers', 'dist_scale',
                'cf_horizon', 'cf_n_rollouts', 'n_score_sample', 'score_interval', 'eval_episodes',
                'save_every', 'target_update_freq', 'n_steps_per_update'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['batch_size'] > config['buffer_capacity']:
        raise ValueError('Replay capacity is smaller than the batch')
    instances = {'small', 'default', 'ring12'}
    if variant == 'option_a':
        instances |= {'clustered', 'hub_outliers', 'two_lobes'}
        for key in ('allow_stranding', 'time_windows'):
            if type(config[key]) is not bool:
                raise ValueError(f'{key} must be boolean')
        if config['reward_shape'] not in ('stepwise', 'terminal'):
            raise ValueError('Unknown reward shape')
        if type(config['n_windowed']) is not int or config['n_windowed'] < 0:
            raise ValueError('n_windowed must be nonnegative')
        if type(config['window_width']) is not int or config['window_width'] < 1:
            raise ValueError('window_width must be positive')
    if config['instance'] not in instances:
        raise ValueError('Unknown instance for this source variant')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    if config['priority_mixing'] not in ('additive', 'multiplicative'):
        raise ValueError('Unknown priority rule')
    if config['consequence_metric'] not in ('total_variation', 'wasserstein', 'kl_divergence', 'jensen_shannon'):
        raise ValueError('Unknown metric')
    if config['consequence_aggregation'] not in ('mean', 'weighted_mean', 'max'):
        raise ValueError('Unknown aggregation')
    return config


def build_agent(config):
    validate_config(config)
    name = 'consequence_dqn' if config['algorithm'] == 'consequence-dqn' else 'dqn'
    module = import_module(f'{__package__}.{config["learner_variant"]}.{name}')
    cls = module.CVRPConsequenceDQN if name == 'consequence_dqn' else module.CVRPDQN
    return cls(deepcopy(config))


def smoke_overrides(config):
    validate_config(config)
    # Task, reward, windows, stranding, budget and source variant stay unchanged.
    return dict(n_episodes=2, batch_size=2, buffer_capacity=16, hidden_dim=8, n_layers=1,
                cf_horizon=2, cf_n_rollouts=2, n_score_sample=2, score_interval=1,
                eval_episodes=1, eval_interval=1, save_every=2, n_checkpoints=0,
                n_steps_per_update=1, target_update_freq=2)


def simulator_sources(config):
    package = import_module(f'experiments.envs.cvrp_budget.{config["learner_variant"]}')
    root = Path(package.__file__).parent
    return {p.name: p for p in root.glob('*.py')}


def evaluation_axis(config):
    return 'episode'
