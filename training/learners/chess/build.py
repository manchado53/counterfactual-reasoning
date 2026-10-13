"""Historical protocol. Validation imports no simulator."""
from copy import deepcopy
from pathlib import Path
import math
from .config import DEFAULT_CHESS_CONFIG as DEFAULTS


def validate_config(config):
    expected = set(DEFAULTS) | {'collector_variant', 'training_budget_kind'}
    if set(config) != expected:
        raise ValueError(f'Unknown: {sorted(set(config)-expected)}; missing: {sorted(expected-set(config))}')
    if config['collector_variant'] != 'historical' or config['training_budget_kind'] != 'chunks':
        raise ValueError('Only the historical protocol is ported')
    if config['algorithm'] not in ('dqn-uniform', 'dqn', 'consequence-dqn'):
        raise ValueError('Unsupported algorithm')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    for key in ['B', 'M', 'hidden_dim', 'cf_horizon', 'cf_n_rollouts', 'n_score_sample', 'score_interval', 'eval_episodes', 'save_every', 'n_chunks', 'C', 'n_steps_for_Q_update', 'cf_top_k', 'n_envs', 'collect_steps']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['B'] > config['M']:
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
    if config['env_name'] != 'gardner_chess' or config['eval_opponent'] not in ('random', 'baseline'):
        raise ValueError('Unsupported chess task/opponent')
    return config


def smoke_overrides(config):
    validate_config(config)
    return {'B': 2, 'M': 16, 'hidden_dim': 8, 'cf_horizon': 2, 'cf_n_rollouts': 2, 'n_score_sample': 2, 'score_interval': 1, 'eval_episodes': 1, 'eval_interval': 1, 'n_checkpoints': 0, 'save_every': 2, 'n_chunks': 2, 'n_envs': 1, 'collect_steps': 2, 'C': 2, 'n_steps_for_Q_update': 1, 'cf_top_k': 2}


def evaluation_axis(config):
    # Chess logs the 1-based collection chunk in the old episode column.
    return 'chunk'


def simulator_sources(config):
    from training.envs import chess
    result = {'chess.py': Path(chess.__file__)}
    import pgx
    root = Path(pgx.__file__).resolve().parent
    result.update({'pgx/' + str(p.relative_to(root)): p for p in root.rglob('*.py')})
    return result


def build_agent(config):
    validate_config(config)
    config = deepcopy(config)
    from .train import create_chess_env
    from .dqn import ChessDQN
    from .consequence_dqn import ChessConsequenceDQN
    env, _, info = create_chess_env(config['seed'], opponent='random')
    return (ChessConsequenceDQN if config['algorithm'] == 'consequence-dqn' else ChessDQN)(env, info, config)
