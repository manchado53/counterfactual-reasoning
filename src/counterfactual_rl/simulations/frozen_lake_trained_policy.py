"""
Record GIFs of a trained FrozenLake agent using the greedy policy.

Usage:
    python -m counterfactual_rl.simulations.frozen_lake_trained_policy --checkpoint <path/to/best.pkl>
    python -m counterfactual_rl.simulations.frozen_lake_trained_policy --checkpoint <path> --episodes 6 --fps 3
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np

from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN
from counterfactual_rl.simulations.frozen_lake_random_policy import render_frame, save_gif


def run_greedy_episode(agent: FrozenLakeDQN, max_steps: int = 200):
    """Run one greedy episode. Returns trajectory list."""
    agent._key, rk = jax.random.split(agent._key)
    _, state = agent.env.reset(rk)
    state = int(state)

    trajectory = [(state, None, 0.0, False)]

    for _ in range(max_steps):
        action = int(agent._greedy_action(agent.params, jnp.int32(state)))
        agent._key, sk = jax.random.split(agent._key)
        _, next_state, reward, done, _ = agent.env.step(
            sk, jnp.int32(state), jnp.int32(action)
        )
        next_state = int(next_state)
        reward = float(reward)
        done = bool(done)
        trajectory.append((next_state, action, reward, done))
        state = next_state
        if done:
            break

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to best.pkl or last.pkl')
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load config from checkpoint first so labels are correct
    import pickle
    with open(args.checkpoint, 'rb') as f:
        ckpt_config = pickle.load(f)['config']

    agent = FrozenLakeDQN({**ckpt_config, 'seed': args.seed})
    agent.load(args.checkpoint)
    agent._key = jax.random.PRNGKey(args.seed)

    alg = ckpt_config.get('algorithm', 'unknown')
    map_name = ckpt_config.get('map_name', '4x4')
    mixing = ckpt_config.get('priority_mixing', '')
    mu = ckpt_config.get('mu', '')

    label = alg
    if alg == 'consequence-dqn':
        if mu == 1.0:
            label = 'cce-only'
        else:
            label = f'cce-{mixing}'

    job_id = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    run_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..', 'runs',
        f"fl_{map_name}_{label}_job{job_id}"
    ))
    os.makedirs(run_dir, exist_ok=True)

    print(f"Algorithm : {alg} ({label})")
    print(f"Map       : {map_name}  Slippery: {agent.config.get('is_slippery', True)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output    : {run_dir}\n")

    outcomes = []
    for ep in range(args.episodes):
        traj = run_greedy_episode(agent, args.max_steps)
        final_state, _, final_reward, _ = traj[-1]
        final_tile = agent.env.desc[divmod(final_state, agent.env.ncols)[0]][divmod(final_state, agent.env.ncols)[1]]
        won = final_tile == 'G'
        outcomes.append({'steps': len(traj) - 1, 'won': won, 'reward': final_reward})

        gif_path = os.path.join(run_dir, f"episode_{ep:02d}.gif")
        save_gif(agent.env, traj, gif_path, fps=args.fps)

        status = "GOAL" if won else ("HOLE" if final_tile == 'H' else "TIMEOUT")
        print(f"  Episode {ep:2d}: {status:7s}  steps={len(traj)-1:3d}  reward={final_reward:.1f}  → {gif_path}")

    win_rate = sum(o['won'] for o in outcomes) / len(outcomes)
    avg_steps = sum(o['steps'] for o in outcomes) / len(outcomes)
    print(f"\nWin rate: {win_rate:.1%}  |  Avg steps: {avg_steps:.1f}")

    summary = (
        f"Algorithm: {alg} ({label})\n"
        f"Map: {map_name}\n"
        f"Slippery: {agent.config.get('is_slippery', True)}\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Episodes: {args.episodes}\n"
        f"Win rate: {win_rate:.1%}\n"
        f"Avg steps: {avg_steps:.1f}\n"
    )
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(summary)

    print(f"Done. GIFs saved to {run_dir}/")


if __name__ == '__main__':
    main()
