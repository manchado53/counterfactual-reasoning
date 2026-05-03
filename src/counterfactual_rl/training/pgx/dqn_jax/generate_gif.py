"""
Generate a GIF of a Gardner chess game played by a trained (or random) agent.

Usage:
    # Random agent (no training, instant):
    python -m counterfactual_rl.training.pgx.dqn_jax.generate_gif --random

    # Train briefly then play:
    python -m counterfactual_rl.training.pgx.dqn_jax.generate_gif --episodes 200

    # Load a saved checkpoint:
    python -m counterfactual_rl.training.pgx.dqn_jax.generate_gif --load path/to/last.pkl

Output: gardner_chess.gif (and gardner_chess.svg) in current directory.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pgx

from .chess_env import GardnerChessEnv, CHESS_ACTIONS
from .dqn import ChessDQN


def play_game(agent: ChessDQN, seed: int = 99, max_steps: int = 200):
    """
    Play one greedy game and return the sequence of pgx states.

    Returns:
        states: list[pgx.State] — one per half-move (both white and black),
                starting from the initial position.
    """
    env = agent.env
    key = jax.random.PRNGKey(seed)

    saved_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy

    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    masks = env.get_legal_mask(state)

    # Store the raw pgx env for frame-by-frame recording
    pgx_env = pgx.make("gardner_chess")
    pgx_rng = jax.random.PRNGKey(seed + 1)

    # Re-play the game step by step, recording every half-move state
    raw_state = pgx_env.init(reset_key)
    states = [raw_state]

    done = False
    steps = 0
    while not done and steps < max_steps:
        action = agent.select_action(obs, masks)
        white_action = int(action[0])

        # Record white's move
        raw_state = pgx_env.step(raw_state, jnp.int32(white_action))
        states.append(raw_state)
        white_done = bool(np.array(raw_state.terminated | raw_state.truncated))

        if white_done:
            break

        # Opponent's move
        pgx_rng, opp_key = jax.random.split(pgx_rng)
        legal = raw_state.legal_action_mask
        black_action = jax.random.choice(
            opp_key,
            jnp.where(legal, jnp.arange(CHESS_ACTIONS), CHESS_ACTIONS),
            p=(legal.astype(jnp.float32) / legal.sum()),
        )
        raw_state = pgx_env.step(raw_state, black_action)
        states.append(raw_state)
        opp_done = bool(np.array(raw_state.terminated | raw_state.truncated))

        # Advance wrapper env too
        next_obs, next_state, reward, done = env.step(state, white_action)
        obs = next_obs
        state = next_state
        masks = env.get_legal_mask(state)
        done = done or opp_done
        steps += 1

    agent.epsilon = saved_eps
    return states


def states_to_gif(states, out_path: str, frame_duration: float = 0.8,
                  scale: float = 2.0):
    """
    Convert a list of pgx.State objects to an animated GIF.

    Strategy:
        1. Save each frame as SVG via pgx.save_svg
        2. Convert SVG → PNG via ImageMagick convert
        3. Combine PNGs → GIF via ImageMagick convert
    """
    out_path = Path(out_path)
    delay_cs = int(frame_duration * 100)  # ImageMagick uses centiseconds

    with tempfile.TemporaryDirectory() as tmpdir:
        png_paths = []
        for i, state in enumerate(states):
            svg_path = os.path.join(tmpdir, f"frame_{i:04d}.svg")
            png_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            pgx.save_svg(state, svg_path, scale=scale)
            subprocess.run(
                ["convert", "-background", "white", "-flatten",
                 svg_path, png_path],
                check=True, capture_output=True,
            )
            png_paths.append(png_path)

        # Assemble GIF
        cmd = ["convert", "-delay", str(delay_cs), "-loop", "0"] + png_paths + [str(out_path)]
        subprocess.run(cmd, check=True, capture_output=True)

    print(f"GIF saved to {out_path}  ({len(states)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random',   action='store_true',
                        help='Skip training, use random policy')
    parser.add_argument('--episodes', type=int, default=0,
                        help='Episodes to train before playing (0 = random)')
    parser.add_argument('--load',     type=str, default=None,
                        help='Path to a saved checkpoint (.pkl)')
    parser.add_argument('--out',      type=str, default='gardner_chess.gif',
                        help='Output GIF path')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--scale',    type=float, default=2.0,
                        help='SVG scale factor (controls GIF size)')
    parser.add_argument('--fps',      type=float, default=1.5,
                        help='Frames per second in output GIF')
    args = parser.parse_args()

    env = GardnerChessEnv(seed=args.seed)
    env_info = {
        'obs_dim': 2875, 'num_agents': 1, 'actions_per_agent': 1225,
        'agent_names': ['white'], 'obs_type': 'flat', 'env_name': 'gardner_chess',
    }

    agent = ChessDQN(env, env_info, config={
        'seed': args.seed,
        'n_episodes': args.episodes,
        'eval_interval': None,
        'M': max(500, args.episodes * 20),
        'B': 32,
    })

    if args.load:
        print(f"Loading checkpoint from {args.load}")
        agent.load(args.load)
    elif args.episodes > 0:
        print(f"Training for {args.episodes} episodes...")
        agent.learn(n_episodes=args.episodes, verbose=True)
    else:
        print("Using random policy (epsilon=1.0)")
        agent.epsilon = 1.0

    print("Playing one game...")
    states = play_game(agent, seed=args.seed)
    print(f"Game finished in {len(states)} half-moves")

    # SVG animation (free, no conversion needed)
    svg_path = str(Path(args.out).with_suffix('.svg'))
    pgx.save_svg_animation(states, svg_path,
                           frame_duration_seconds=1.0 / args.fps)
    print(f"SVG animation saved to {svg_path}")

    # GIF
    states_to_gif(states, args.out,
                  frame_duration=1.0 / args.fps,
                  scale=args.scale)


if __name__ == '__main__':
    main()
