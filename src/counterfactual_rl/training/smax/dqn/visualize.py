"""Visualize trained DQN agent playing SMAX."""

import argparse
from .dqn import DQN
from .utils import record_episode, save_gameplay_gif


def main():
    """Record and save gameplay of trained agent."""
    parser = argparse.ArgumentParser(description='Visualize trained DQN agent')
    parser.add_argument('--checkpoint', type=str, default='models/smax_dqn_3m.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='gameplay.gif',
                        help='Output GIF path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for episode')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to record')
    parser.add_argument('--greedy', action='store_true', default=True,
                        help='Use greedy policy (no exploration)')
    parser.add_argument('--no-greedy', dest='greedy', action='store_false',
                        help='Use epsilon-greedy policy')
    args = parser.parse_args()

    # Load trained agent
    print(f"Loading checkpoint: {args.checkpoint}")
    agent = DQN.from_checkpoint(args.checkpoint)
    print(f"  Scenario: {agent.env_info['scenario']}")
    print(f"  Num agents: {agent.num_agents}")

    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"\nRecording episode {ep + 1}/{args.episodes} (seed={seed})...")

        # Record episode
        state_seq, episode_return, episode_length = record_episode(
            agent, seed=seed, greedy=args.greedy
        )
        print(f"  Return: {episode_return:.2f}, Length: {episode_length}")

        # Save GIF
        if args.episodes == 1:
            output_path = args.output
        else:
            base, ext = args.output.rsplit('.', 1)
            output_path = f"{base}_{ep + 1}.{ext}"

        print(f"  Saving to: {output_path}")
        save_gameplay_gif(agent.env, state_seq, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
