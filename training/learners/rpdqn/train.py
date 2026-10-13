"""
CLI entry point for the neural-CVRP (RP-DQN) agent.

    python -m counterfactual_rl.agents.cvrp_neural.train \
        --algorithm dqn --episodes 2000 --seed 0

Config precedence (same as every other agent here):
    DEFAULT_CONFIG  <  CLI flags  <  CONFIG_OVERRIDES_B64

CONFIG_OVERRIDES_B64 is applied LAST on purpose -- the SLURM sweep drivers pass
the entire arm config through it and must be able to override a CLI flag.
"""

import argparse
import ast
import base64
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', default=None,
                   choices=['dqn-uniform', 'dqn', 'consequence-dqn'],
                   help='Algorithm variant (default from config)')
    p.add_argument('--episodes', type=int, default=None, dest='n_episodes')
    p.add_argument('--customers', type=int, default=None, dest='n_customers',
                   help='Problem size n (20 / 50 / 100)')
    p.add_argument('--mixing', default=None, choices=['additive', 'multiplicative'],
                   dest='priority_mixing',
                   help='Consequence+TD mixing mode (consequence-dqn only)')
    p.add_argument('--mu', type=float, default=None,
                   help='Additive mixing weight mu (consequence-dqn + additive only)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--override', action='append', default=[], metavar='KEY=VALUE',
                   help='Override any config key, e.g. --override cf_horizon=5')
    return p.parse_args()


def build_config(args) -> dict:
    from .config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()

    for key in ('algorithm', 'n_episodes', 'n_customers', 'priority_mixing', 'mu', 'seed'):
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    for item in args.override:
        k, _, v = item.partition('=')
        try:
            config[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            config[k] = v

    # Applied last so a sweep's base64 blob wins over anything above.
    env_b64 = os.environ.get('CONFIG_OVERRIDES_B64')
    if env_b64:
        config.update(json.loads(base64.b64decode(env_b64).decode()))

    return config


def main():
    args = parse_args()
    config = build_config(args)

    if config['algorithm'] == 'consequence-dqn':
        if config.get('vectorized', True):
            from .consequence_dqn_vectorized import CVRPNeuralConsequenceDQNVectorized as Agent
        else:
            from .consequence_dqn import CVRPNeuralConsequenceDQN as Agent
    else:
        if config.get('vectorized', True):
            from .dqn_vectorized import CVRPNeuralDQNVectorized as Agent
        else:
            from .dqn import CVRPNeuralDQN as Agent

    agent = Agent(config)
    agent.learn()


if __name__ == '__main__':
    main()
