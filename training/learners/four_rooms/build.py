"""Lazy adapter; validation loads no JAX runtime."""
from copy import deepcopy
from pathlib import Path

ARMS = ('uniform', 'rank_per', 'cce')
SUPPORTED_ARMS = ARMS + ('proportional_per',)


def config_for(arm, seed, room_scale=1):
    if type(room_scale) is not int or room_scale not in (1, 2):
        raise ValueError("Supported room scales are 1 and 2")
    config = dict(arm=arm, seed=seed, gamma=.99, alpha=.0003, hidden_dim=128, n_layers=3,
        batch_size=128, buffer_capacity=100000, n_steps_per_update=16, target_update_freq=2000,
        n_envs=32, collect_steps=32, warmup_steps=1024, blocks=512, eval_every_blocks=16,
        epsilon_start=1., epsilon_end=.05, epsilon_decay_steps=262144,
        fixed_eval_seed=20261015, eval_episodes=1000, cf_horizon=60, cf_n_rollouts=1,
        cf_gamma=.99, cf_rollout_temperature=0., cf_action_prob_temperature=.5,
        cf_bootstrap=True, score_on_insert=True, normalise_priorities='rank',
        priority_mixing='multiplicative', mu=.5, mu_c=1., mu_delta=1.,
        PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1.),
        protocol='sparse-four-rooms-v1', training_budget_kind='env_steps',
        early_stop_win_rate=None)
    if room_scale != 1:
        config.update(room_scale=room_scale, protocol='sparse-four-rooms-2x-v1')
    if arm == 'proportional_per':
        # Shared Eq. 2 with raw TD, and Eq. 4 with zero CCE contribution.
        config.update(normalise_priorities='none', priority_mixing='additive', mu=0.)
    return config


def smoke_overrides(config):
    return dict(n_envs=4, collect_steps=32, buffer_capacity=512, warmup_steps=128,
                blocks=3, eval_every_blocks=1, eval_episodes=8)


def validate_config(config):
    if config.get('arm') not in SUPPORTED_ARMS or type(config.get('seed')) is not int or config['seed'] not in range(5):
        raise ValueError('Unknown arm or seed')
    expected = config_for(config['arm'], config['seed'], config.get('room_scale', 1))
    smoke = dict(expected, **smoke_overrides(expected))
    if config != expected and config != smoke:
        raise ValueError('Config differs from frozen full or smoke protocol')
    return config


def build_agent(config):
    validate_config(config)
    from training.envs.four_rooms import FourRoomsEnv
    from training.learners.shared.discrete import DiscreteAgent
    return DiscreteAgent(deepcopy(config), FourRoomsEnv(config.get('room_scale', 1)))


def evaluation_axis(config):
    return 'env_steps'


def simulator_sources(config):
    from training.envs import four_rooms
    return {'four_rooms.py': Path(four_rooms.__file__)}
