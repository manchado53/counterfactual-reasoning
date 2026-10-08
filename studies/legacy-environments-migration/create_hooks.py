"""Generate explicit historical adapters and recipes using apply_patch."""
import json
import runpy
from pathlib import Path
from port_sources import ROOT, add

COMMON = '''"""Historical protocol. Validation imports no simulator."""
from copy import deepcopy
from pathlib import Path
import math
from .config import DEFAULT_NAME as DEFAULTS


def validate_config(config):
    expected = set(DEFAULTS) | {'collector_variant', 'training_budget_kind'}
    if set(config) != expected:
        raise ValueError(f'Unknown: {sorted(set(config)-expected)}; missing: {sorted(expected-set(config))}')
    if config['collector_variant'] != 'historical' or config['training_budget_kind'] != BUDGET:
        raise ValueError('Only the historical protocol is ported')
    if config['algorithm'] not in ('dqn-uniform', 'dqn', 'consequence-dqn'):
        raise ValueError('Unsupported algorithm')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    for key in POSITIVE:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config[BATCH] > config[CAPACITY]:
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
    EXTRA_VALIDATION
    return config


def smoke_overrides(config):
    validate_config(config)
    return SMOKE


def evaluation_axis(config):
    # Chess logs the 1-based collection chunk in the old episode column.
    return AXIS


def simulator_sources(config):
    from experiments.envs import ENV
    result = {'ENV.py': Path(ENV.__file__)}
    SIMULATOR
    return result


def build_agent(config):
    validate_config(config)
    config = deepcopy(config)
    BUILD
'''


def main():
    files = {}
    for env in ('smax', 'chess', 'cvrp'):
        config_path = ROOT / f'experiments/learners/{env}/config.py'
        default_name = 'DEFAULT_CHESS_CONFIG' if env == 'chess' else 'DEFAULT_CONFIG'
        cfg = runpy.run_path(str(config_path))[default_name]
        budget = 'chunks' if env == 'chess' else 'episodes'
        cfg.update(collector_variant='historical', training_budget_kind=budget)
        batch, capacity = ('batch_size', 'buffer_capacity') if env == 'cvrp' else ('B', 'M')
        positive = [batch, capacity, 'hidden_dim', 'cf_horizon', 'cf_n_rollouts', 'n_score_sample',
                    'score_interval', 'eval_episodes', 'save_every', 'n_chunks' if env == 'chess' else 'n_episodes']
        smoke = {batch: 2, capacity: 16, 'hidden_dim': 8, 'cf_horizon': 2, 'cf_n_rollouts': 2,
                 'n_score_sample': 2, 'score_interval': 1, 'eval_episodes': 1, 'eval_interval': 1,
                 'n_checkpoints': 0, 'save_every': 2}
        extra = ''
        if env == 'smax':
            positive += ['C', 'n_steps_for_Q_update', 'cf_top_k', 'n_body_layers', 'n_head_layers']
            smoke.update(n_episodes=2, n_body_layers=1, n_head_layers=1, C=2,
                         n_steps_for_Q_update=1, cf_top_k=2)
            extra = "if config['backend'] != 'jax' or config['obs_type'] not in ('world_state', 'concatenated'):\n        raise ValueError('Unsupported SMAX backend/observation')"
            build = '''from experiments.envs.smax import create_smax_env
    from .dqn import DQN
    from .consequence_dqn import ConsequenceDQN
    from .utils import evaluate
    from types import MethodType
    env, _, info = create_smax_env(config['scenario'], config['seed'], config['obs_type'])
    agent = (ConsequenceDQN if config['algorithm'] == 'consequence-dqn' else DQN)(env, info, config)
    agent.evaluate = MethodType(evaluate, agent)
    return agent'''
        elif env == 'chess':
            positive += ['C', 'n_steps_for_Q_update', 'cf_top_k', 'n_envs', 'collect_steps']
            smoke.update(n_chunks=2, n_envs=1, collect_steps=2, C=2, n_steps_for_Q_update=1, cf_top_k=2)
            extra = "if config['env_name'] != 'gardner_chess' or config['eval_opponent'] not in ('random', 'baseline'):\n        raise ValueError('Unsupported chess task/opponent')"
            build = '''from .train import create_chess_env
    from .dqn import ChessDQN
    from .consequence_dqn import ChessConsequenceDQN
    env, _, info = create_chess_env(config['seed'], opponent='random')
    return (ChessConsequenceDQN if config['algorithm'] == 'consequence-dqn' else ChessDQN)(env, info, config)'''
        else:
            positive += ['target_update_freq', 'n_steps_per_update', 'n_layers']
            smoke.update(n_episodes=2, target_update_freq=2, n_steps_per_update=1, n_layers=1)
            extra = "if config['vectorized'] is not False or config['instance'] not in ('default', 'small'):\n        raise ValueError('Only historical tabular CVRP/TSP is ported')\n    if config['capacity'] is not None and (type(config['capacity']) is not int or config['capacity'] < 1):\n        raise ValueError('Invalid capacity')\n    if not math.isfinite(config['travel_noise']) or config['travel_noise'] < 0:\n        raise ValueError('Invalid travel_noise')"
            build = '''from .dqn import CVRPDQN
    from .consequence_dqn import CVRPConsequenceDQN
    return (CVRPConsequenceDQN if config['algorithm'] == 'consequence-dqn' else CVRPDQN)(config)'''
        simulator = 'pass'
        if env != 'cvrp':
            package = 'jaxmarl' if env == 'smax' else 'pgx'
            simulator = f"import {package}\n    root = Path({package}.__file__).resolve().parent\n    result.update({{'{package}/' + str(p.relative_to(root)): p for p in root.rglob('*.py')}})"
        text = COMMON.replace('DEFAULT_NAME', default_name)
        for key, val in [('BUDGET', repr(budget)), ('POSITIVE', repr(positive)), ('BATCH', repr(batch)),
                         ('CAPACITY', repr(capacity)), ('EXTRA_VALIDATION', extra), ('SMOKE', repr(smoke)),
                         ('AXIS', repr('chunk' if env == 'chess' else 'episode')), ('SIMULATOR', simulator),
                         ('BUILD', build), ('ENV', env)]:
            text = text.replace(key, val)
        files[Path(f'experiments/learners/{env}/build.py')] = text
        rows = []
        for task, algorithm in enumerate(('dqn-uniform', 'dqn', 'consequence-dqn'), 1):
            row_cfg = dict(cfg, algorithm=algorithm)
            rows.append(dict(task=task, seed=cfg['seed'], collector='historical', config=row_cfg))
        files[Path(f'configs/experiment/{env}-historical.json')] = json.dumps(
            dict(schema_version=1, environment=env, rows=rows), indent=2) + '\n'
    add(files)


if __name__ == '__main__':
    main()
