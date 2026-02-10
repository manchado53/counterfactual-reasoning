"""Entry point for training DQN on SMAX."""

import argparse
from .dqn import DQN
from .utils import create_smax_env


def main():
    """Train DQN agent on SMAX."""
    parser = argparse.ArgumentParser(description='Train DQN on SMAX')
    parser.add_argument('--scenario', type=str, default='3m',
                        help='SMAX scenario (default: 3m)')
    parser.add_argument('--n-episodes', type=int, default=2000,
                        help='Number of training episodes (default: 2000)')
    parser.add_argument('--save-path', type=str, default='models/smax_dqn.pt',
                        help='Path to save model (default: models/smax_dqn.pt)')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save checkpoint every N episodes (default: 500)')
    parser.add_argument('--plot-path', type=str, default='training_curves.png',
                        help='Path to save training curves (default: training_curves.png)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    args = parser.parse_args()

    # Create environment
    env, key, env_info = create_smax_env(scenario=args.scenario, seed=args.seed)

    # Create and train agent
    agent = DQN(env, env_info, config={
        'n_episodes': args.n_episodes,
        'save_path': args.save_path,
        'save_every': args.save_every,
    })

    agent.learn()
    agent.plot_training_curves(save_path=args.plot_path, show=False)


if __name__ == '__main__':
    main()
