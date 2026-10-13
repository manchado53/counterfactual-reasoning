"""Lazy runner hooks for the pinned fixed-opponent transfer."""
from copy import deepcopy
from pathlib import Path
from .config import ARMS, config_for


def resolved_config(arm, seed):
    return dict(config_for(arm, seed), protocol='fixed-jaxnav-transfer',
                collector_variant='fixed', training_budget_kind='env_steps',
                early_stop_win_rate=None)


def validate_config(config):
    if config.get('protocol') == 'legacy-connect-four':
        from .legacy.build import validate_config as validate_legacy
        return validate_legacy(config)
    if config.get('arm') not in ARMS or type(config.get('seed')) is not int or config['seed'] not in (0, 1):
        raise ValueError('Pinned transfer requires a source arm and seed 0 or 1')
    expected = resolved_config(config['arm'], config['seed'])
    adjustable = {'n_envs', 'collect_steps', 'buffer_capacity', 'blocks', 'eval_every_blocks'}
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
    if config['protocol'] == 'legacy-connect-four':
        return dict(n_envs=2,collect_steps=8,n_chunks=2,M=64,B=4,hidden_dim=8,
                    eval_episodes=8,eval_interval=1,save_every=1,n_checkpoints=0,
                    score_interval=1,n_score_sample=2,cf_horizon=4,cf_n_rollouts=2)
    return dict(n_envs=4, collect_steps=32, buffer_capacity=256, blocks=2, eval_every_blocks=1)


def evaluation_axis(config):
    return 'env_steps'


def simulator_sources(config):
    import pgx
    root = Path(pgx.__file__).resolve().parent
    sources = {'pgx/' + str(p.relative_to(root)): p for p in root.rglob('*.py')}
    if config['protocol'] == 'fixed-jaxnav-transfer':
        from training.envs import connect_four_transfer
        sources['connect_four_transfer.py'] = Path(connect_four_transfer.__file__)
    elif config['opponent'] == 'mcts':
        import mctx
        root = Path(mctx.__file__).resolve().parent
        sources.update({'mctx/'+str(p.relative_to(root)): p for p in root.rglob('*.py')})
    return sources


def build_agent(config):
    validate_config(config)
    if config['protocol'] == 'legacy-connect-four':
        from .legacy.build import build_agent as build_legacy
        return build_legacy(deepcopy(config))
    from .runner import Agent
    return Agent(deepcopy(config))
