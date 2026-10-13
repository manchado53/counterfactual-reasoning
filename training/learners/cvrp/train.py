"""
Entry point for routing (CVRP / TSP) DQN training.

Usage:
    python -m counterfactual_rl.agents.cvrp.train
    python -m counterfactual_rl.agents.cvrp.train --algorithm dqn-uniform
    python -m counterfactual_rl.agents.cvrp.train --algorithm consequence-dqn --mixing multiplicative
    python -m counterfactual_rl.agents.cvrp.train --capacity 6 --seed 3

Algorithms:
    dqn-uniform      Vanilla DQN (uniform buffer)
    dqn              DQN + PER
    consequence-dqn  DQN + CCE (mixing controlled by --mixing / --mu)

Routing-specific flags:
    --capacity N     the load limit (the "C" in CVRP); -1 selects TSP mode.
                     This is the CAPACITY DIAL: tighter capacity -> more pivotal
                     reload decisions -> higher stakes-concentration.
    --travel-noise F traffic. 0.0 (default) = deterministic, for Claim 2. Use > 0 for
                     Claim 1, where the total-variation score needs return variation.

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
                   choices=['dqn-uniform', 'dqn', 'consequence-dqn'])
    p.add_argument('--instance', default=None, choices=['default', 'small'])
    p.add_argument('--capacity', type=int, default=None,
                   help='load limit; -1 for TSP mode (no limit)')
    p.add_argument('--travel-noise', type=float, default=None, dest='travel_noise')
    p.add_argument('--episodes', type=int, default=None, dest='n_episodes')
    p.add_argument('--mixing', default=None, choices=['additive', 'multiplicative'],
                   dest='priority_mixing')
    p.add_argument('--mu', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--override', action='append', default=[], metavar='KEY=VALUE')
    return p.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    if args.algorithm is not None:
        config['algorithm'] = args.algorithm
    if args.instance is not None:
        config['instance'] = args.instance
    if args.capacity is not None:
        config['capacity'] = None if args.capacity < 0 else args.capacity
    if args.travel_noise is not None:
        config['travel_noise'] = args.travel_noise
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

    if config['algorithm'] == 'consequence-dqn':
        from .consequence_dqn import CVRPConsequenceDQN
        agent = CVRPConsequenceDQN(config)
    else:
        from .dqn import CVRPDQN
        agent = CVRPDQN(config)

    agent.learn()


if __name__ == '__main__':
    main()
