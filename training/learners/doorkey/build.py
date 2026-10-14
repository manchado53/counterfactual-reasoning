"""Lazy hooks; fixed transfer and historical episode collectors stay explicit."""
from copy import deepcopy
import math
from pathlib import Path
from .config import (
    ARMS,
    config_for,
    DECAY_TRANSFER_ARMS,
    decay_transfer_config,
    PROCEDURAL_BENCHMARK_ARMS,
    procedural_benchmark_config,
)


def resolved_config(arm, seed):
    return dict(config_for(arm, seed), protocol='fixed-jaxnav-transfer',
                collector_variant='fixed', training_budget_kind='env_steps',
                early_stop_win_rate=None)


def validate_config(config):
    if config.get('protocol') == 'procedural-doorkey-12x12':
        if config.get('arm') not in PROCEDURAL_BENCHMARK_ARMS:
            raise ValueError('Unsupported procedural DoorKey arm')
        expected = procedural_benchmark_config(config['arm'], config.get('seed'))
        if config.get('_smoke_profile') is True:
            expected.update(smoke_overrides(expected))
        if config != expected:
            changed = sorted(key for key in set(config) | set(expected)
                             if config.get(key) != expected.get(key))
            raise ValueError(f'Procedural DoorKey config drift: {changed}')
        return config
    if config.get('protocol') == 'jaxnav-ranked-td-decay-transfer':
        if config.get('arm') not in DECAY_TRANSFER_ARMS:
            raise ValueError('Unsupported DoorKey decay-transfer arm')
        expected = decay_transfer_config(
            config['arm'], config.get('seed'), config.get('layout_name'))
        if config.get('_smoke_profile') is True:
            expected.update(
                n_envs=4,
                collect_steps=32,
                buffer_capacity=256,
                n_episodes=8,
                eval_interval=4,
                save_every=8,
                decay_start=2,
                decay_end=4,
                _smoke_profile=True,
            )
        if config != expected:
            changed = sorted(key for key in set(config) | set(expected)
                             if config.get(key) != expected.get(key))
            raise ValueError(f'DoorKey JaxNav-transfer config drift: {changed}')
        return config
    if config.get('protocol') == 'legacy-doorkey':
        from .legacy.config import DEFAULT_CONFIG
        required = set(DEFAULT_CONFIG) | {'protocol', 'collector_variant', 'training_budget_kind', 'early_stop_win_rate'}
        if set(config) != required:
            raise ValueError('Unknown or unresolved historical settings')
        if config['collector_variant'] != ('old' if config['vectorized'] else 'sequential') or config['training_budget_kind'] != 'episodes':
            raise ValueError('Historical collector metadata mismatch')
        if config['layout_name'] not in ('6x6', '6x6_lava','12x12','16x16','12x12_lava','16x16_lava') or config['algorithm'] not in ('dqn-uniform','dqn','consequence-dqn'):
            raise ValueError('Unsupported historical task or learner')
        for key in ('n_envs','collect_steps','batch_size','buffer_capacity','n_episodes','max_episode_steps',
                    'eval_interval','eval_episodes','save_every','target_update_freq','n_steps_per_update',
                    'hidden_dim','n_layers','cf_horizon','cf_n_rollouts','score_interval','n_score_sample','epsilon_decay_episodes'):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if config['vectorized'] and config['collect_steps'] < config['max_episode_steps']:
            raise ValueError('Old collector resets episode length each chunk; timeout must fit')
        if config['batch_size'] > config['buffer_capacity']:
            raise ValueError('Batch exceeds replay capacity')
        if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
            raise ValueError('Invalid seed')
        if type(config['vectorized']) is not bool or type(config['n_checkpoints']) is not int or config['n_checkpoints'] < 0:
            raise ValueError('Invalid collector/checkpoint settings')
        for key in ('gamma','cf_gamma','mu','epsilon_start','epsilon_end'):
            if type(config[key]) not in (int,float) or not 0 <= config[key] <= 1:
                raise ValueError(f'Invalid {key}')
        for key in ('alpha','mu_c','mu_delta'):
            if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or config[key] < 0:
                raise ValueError(f'Invalid {key}')
        if config['early_stop_win_rate'] is not None and (type(config['early_stop_win_rate']) not in (int,float) or not 0 <= config['early_stop_win_rate'] <= 1):
            raise ValueError('Invalid stopping threshold')
        if set(config['PER_parameters']) != {'eps','beta','maximum_priority'}:
            raise ValueError('Unknown or missing replay parameters')
        if config['priority_mixing'] not in ('additive','multiplicative'):
            raise ValueError('Invalid priority mixing')
        if not 0 <= config['slip_prob'] <= 1 or config['random_start'] not in ('fixed','agent','exploring') or config['eval_random_start'] not in ('fixed','agent','exploring'):
            raise ValueError('Invalid transition/reset rules')
        return config
    if config.get('arm') not in ARMS or type(config.get('seed')) is not int or config['seed'] not in (0,1):
        raise ValueError('Pinned transfer requires a source arm and seed 0 or 1')
    expected = resolved_config(config['arm'], config['seed'])
    adjustable = {'n_envs','collect_steps','buffer_capacity','blocks','eval_every_blocks'}
    if set(config) != set(expected):
        raise ValueError('Unknown or unresolved transfer settings')
    for key, value in expected.items():
        if key not in adjustable and config[key] != value:
            raise ValueError(f'Pinned source setting changed: {key}')
    for key in adjustable:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 128 <= config['n_envs'] * config['collect_steps'] <= config['buffer_capacity']:
        raise ValueError('A collection block must fill a batch and fit replay')
    return config


def smoke_overrides(config):
    if config['protocol'] == 'legacy-doorkey':
        return dict(n_envs=2, collect_steps=max(64,config['max_episode_steps']), buffer_capacity=256, batch_size=4,
                    n_episodes=2, hidden_dim=8, n_layers=1, eval_interval=1,
                    eval_episodes=2, n_checkpoints=0, save_every=2,
                    cf_horizon=4, cf_n_rollouts=2, score_interval=1, n_score_sample=4)
    if config['protocol'] == 'jaxnav-ranked-td-decay-transfer':
        return dict(n_envs=4, collect_steps=32, buffer_capacity=256,
                    n_episodes=8, eval_interval=4, save_every=8,
                    decay_start=2, decay_end=4, _smoke_profile=True)
    if config['protocol'] == 'procedural-doorkey-12x12':
        return dict(
            n_envs=4,
            collect_steps=32,
            buffer_capacity=256,
            batch_size=16,
            total_env_steps=256,
            eval_interval_steps=128,
            eval_episodes=8,
            max_episode_steps=48,
            eval_max_steps=48,
            epsilon_decay_steps=128,
            cf_horizon=4,
            cf_score_chunk=128,
            decay_start=64,
            decay_end=192,
            _smoke_profile=True,
        )
    return dict(n_envs=4, collect_steps=32, buffer_capacity=256, blocks=2, eval_every_blocks=1)


def evaluation_axis(config):
    return config['training_budget_kind'][:-1] if config['training_budget_kind'] == 'episodes' else 'env_steps'


def simulator_sources(config):
    if config['protocol'] == 'procedural-doorkey-12x12':
        name = 'doorkey_procedural.py'
    else:
        name = 'doorkey_legacy.py' if config['protocol'] == 'legacy-doorkey' else 'doorkey.py'
    return {name: Path(__file__).resolve().parents[2] / 'envs' / name}


def build_agent(config):
    validate_config(config)
    config = deepcopy(config)
    if config['protocol'] == 'procedural-doorkey-12x12':
        from .procedural_runner import ProceduralAgent
        return ProceduralAgent(config)
    if config['protocol'] == 'legacy-doorkey':
        if config['vectorized']:
            from .legacy.dqn_vectorized import DoorKeyDQNVectorized as DQN
            from .legacy.consequence_dqn_vectorized import DoorKeyConsequenceDQNVectorized as CCE
        else:
            from .legacy.dqn import DoorKeyDQN as DQN
            from .legacy.consequence_dqn import DoorKeyConsequenceDQN as CCE
        agent = (CCE if config['algorithm'] == 'consequence-dqn' else DQN)(config)
        agent.evaluation_identity = lambda: dict(eval_bank_hash=None,
            eval_rng_mode='consumes-training-key', eval_max_steps=200,
            eval_fixed_bank=False, eval_reset=config['eval_random_start'])
        return agent
    from .runner import Agent
    return Agent(config)
