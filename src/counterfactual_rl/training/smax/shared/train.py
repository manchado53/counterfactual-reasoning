"""Entry point for training DQN on SMAX with backend selection."""

import argparse
from .utils import create_smax_env, evaluate


def main():
    """Train DQN agent on SMAX."""
    parser = argparse.ArgumentParser(description='Train DQN on SMAX')
    parser.add_argument('--backend', type=str, default='jax',
                        choices=['jax', 'pytorch'],
                        help='DQN backend (default: jax)')
    parser.add_argument('--scenario', type=str, default='3m',
                        help='SMAX scenario (default: 3m)')
    parser.add_argument('--n-episodes', type=int, default=2000,
                        help='Number of training episodes (default: 2000)')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save checkpoint every N episodes (default: 500)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--obs-type', type=str, default='world_state',
                        choices=['world_state', 'concatenated'],
                        help='Observation type (default: world_state)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Evaluation episodes after training (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Run greedy evaluation every N episodes during training (default: disabled)')
    args = parser.parse_args()

    # Dynamic import based on backend
    if args.backend == 'jax':
        from ..dqn_jax.dqn import DQN
    else:
        from ..dqn_pytorch.dqn import DQN

    # Create environment
    env, key, env_info = create_smax_env(
        scenario=args.scenario, seed=args.seed, obs_type=args.obs_type
    )

    # Create and train agent
    agent = DQN(env, env_info, config={
        'n_episodes': args.n_episodes,
        'save_every': args.save_every,
        'eval_interval': args.eval_interval,
        'eval_episodes': args.eval_episodes,
    })

    agent.learn()

    # Post-training evaluation
    evaluate(agent, n_episodes=args.eval_episodes, parallel=(args.backend == 'jax'))


if __name__ == '__main__':
    main()
