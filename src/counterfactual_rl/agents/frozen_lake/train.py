"""
Entry point for FrozenLake DQN training.

Usage:
    python -m counterfactual_rl.agents.frozen_lake.train
    python -m counterfactual_rl.agents.frozen_lake.train --algorithm dqn-uniform --map 8x8
    python -m counterfactual_rl.agents.frozen_lake.train --algorithm consequence-dqn --mixing additive

Algorithms:
    dqn-uniform    Vanilla DQN (uniform buffer)
    dqn            DQN + PER
    consequence-dqn  DQN + CCE (mixing controlled by --mixing / --mu flags)

Override any config key via --override KEY=VALUE (repeatable).
"""

import argparse
import ast
import base64
import json
import os

from .config import DEFAULT_CONFIG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', default=None,
                   choices=['dqn-uniform', 'dqn', 'consequence-dqn'],
                   help='Algorithm variant (default from config)')
    p.add_argument('--map', default=None, choices=['4x4', '8x8'],
                   dest='map_name', help='FrozenLake map size')
    p.add_argument('--no-slippery', action='store_true', default=None,
                   help='Deterministic transitions')
    p.add_argument('--episodes', type=int, default=None, dest='n_episodes')
    p.add_argument('--mixing', default=None, choices=['additive', 'multiplicative'],
                   dest='priority_mixing',
                   help='Consequence+TD mixing mode (consequence-dqn only)')
    p.add_argument('--mu', type=float, default=None,
                   help='Additive mixing weight μ (consequence-dqn + additive only)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--override', action='append', default=[], metavar='KEY=VALUE',
                   help='Override any config key, e.g. --override alpha=0.0005')
    return p.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    # CLI overrides
    if args.algorithm is not None:
        config['algorithm'] = args.algorithm
    if args.map_name is not None:
        config['map_name'] = args.map_name
    if args.no_slippery:
        config['is_slippery'] = False
    if args.n_episodes is not None:
        config['n_episodes'] = args.n_episodes
    if args.priority_mixing is not None:
        config['priority_mixing'] = args.priority_mixing
    if args.mu is not None:
        config['mu'] = args.mu
    if args.seed is not None:
        config['seed'] = args.seed

    for kv in args.override:
        key, _, raw_val = kv.partition('=')
        try:
            config[key.strip()] = ast.literal_eval(raw_val.strip())
        except (ValueError, SyntaxError):
            config[key.strip()] = raw_val.strip()

    # Env-var overrides take precedence (used by run_experiments.py batch submission)
    env_b64 = os.environ.get('CONFIG_OVERRIDES_B64')
    if env_b64:
        config.update(json.loads(base64.b64decode(env_b64).decode()))

    # Select and run agent
    if config['algorithm'] == 'consequence-dqn':
        from .consequence_dqn import FrozenLakeConsequenceDQN
        agent = FrozenLakeConsequenceDQN(config)
    else:
        from .dqn import FrozenLakeDQN
        agent = FrozenLakeDQN(config)

    agent.learn()


if __name__ == '__main__':
    main()
