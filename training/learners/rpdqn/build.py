"""Pinned neural-routing RP-DQN; historical dirty source, not upstream-paper parity."""
from copy import deepcopy
from pathlib import Path
import math
from .config import DEFAULT_CONFIG


def validate_config(config):
    required = set(DEFAULT_CONFIG) | {'collector_variant', 'training_budget_kind'}
    if set(config) - required - {'_arm'} or required - set(config):
        raise ValueError(f'Unknown: {sorted(set(config)-required-{"_arm"})}; missing: {sorted(required-set(config))}')
    if config['collector_variant'] != 'historical' or config['training_budget_kind'] != 'episodes':
        raise ValueError('Only the historical episode protocol is ported')
    if config['vectorized'] is not True or config['n_step'] != 1:
        raise ValueError('Only the historical vectorized one-step learner is ported')
    if config['algorithm'] not in ('dqn-uniform', 'dqn', 'consequence-dqn'):
        raise ValueError('Unsupported algorithm')
    for key in ('n_customers', 'max_steps', 'n_episodes', 'n_envs', 'collect_steps',
                'buffer_capacity', 'buffer_min_samples', 'batch_size', 'n_steps_per_update',
                'target_update_freq', 'embed_dim', 'n_layers', 'n_heads', 'ff_dim',
                'eval_episodes', 'save_every', 'cf_horizon', 'cf_n_rollouts', 'score_interval', 'n_score_sample'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be an explicit positive integer')
    if config['capacity'] is None or not math.isfinite(config['capacity']) or config['capacity'] < 9:
        raise ValueError('Explicit capacity >= maximum demand (9) required')
    if max(config['batch_size'], config['buffer_min_samples']) > config['buffer_capacity']:
        raise ValueError('Replay is too small')
    if config['embed_dim'] % config['n_heads']:
        raise ValueError('Heads must divide embedding size')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    if config['exploration'] not in ('boltzmann', 'epsilon'):
        raise ValueError('Unsupported exploration')
    if config['priority_mixing'] not in ('additive', 'multiplicative'):
        raise ValueError('Unsupported priority rule')
    if config['consequence_metric'] not in ('wasserstein', 'total_variation', 'kl_divergence', 'jensen_shannon'):
        raise ValueError('Unsupported metric')
    if config['consequence_aggregation'] not in ('mean', 'max', 'weighted_mean'):
        raise ValueError('Unsupported aggregation')
    if set(config['PER_parameters']) != {'eps', 'beta', 'maximum_priority'}:
        raise ValueError('Unresolved PER settings')
    return config


def build_agent(config):
    validate_config(config)
    from .dqn_vectorized import CVRPNeuralDQNVectorized
    from .consequence_dqn_vectorized import CVRPNeuralConsequenceDQNVectorized
    cls = CVRPNeuralConsequenceDQNVectorized if config['algorithm'] == 'consequence-dqn' else CVRPNeuralDQNVectorized
    return cls(deepcopy(config))


def smoke_overrides(config):
    validate_config(config)
    return dict(n_episodes=2, n_envs=1, collect_steps=4, batch_size=2,
        buffer_capacity=16, buffer_min_samples=2, n_steps_per_update=4,
        embed_dim=8, n_layers=1, n_heads=1, ff_dim=16, eval_interval=1,
        eval_episodes=2, save_every=2, n_checkpoints=0,
        cf_horizon=2, cf_n_rollouts=1, score_interval=1, n_score_sample=2)


def simulator_sources(config):
    from training.envs import rpdqn
    return {'rpdqn.py': Path(rpdqn.__file__)}


def evaluation_axis(config):
    return 'episode'
