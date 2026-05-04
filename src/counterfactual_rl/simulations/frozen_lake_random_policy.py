"""
Random policy rollouts on JAX FrozenLake — records GIFs of each episode.

Usage:
    python -m counterfactual_rl.simulations.frozen_lake_random_policy
    python -m counterfactual_rl.simulations.frozen_lake_random_policy --map 8x8 --episodes 5
    python -m counterfactual_rl.simulations.frozen_lake_random_policy --map 4x4 --no-slippery

Outputs (written to runs/frozen_lake_random_<timestamp>/):
    episode_0.gif ... episode_N.gif
    summary.txt
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv

# ── tile colours ──────────────────────────────────────────────────────────────
TILE_COLORS = {
    'S': '#5DADE2',   # blue  — start
    'F': '#F0F3F4',   # light grey — frozen
    'H': '#1C2833',   # dark — hole
    'G': '#27AE60',   # green — goal
}
TILE_TEXT_COLORS = {
    'S': 'white',
    'F': '#555',
    'H': 'white',
    'G': 'white',
}
AGENT_COLOR = '#E74C3C'   # red


# ── rendering ─────────────────────────────────────────────────────────────────

def render_frame(ax, env, state: int, step: int, reward: float, done: bool,
                 action: int = None):
    ax.clear()
    desc = env.desc
    nrows, ncols = env.nrows, env.ncols

    action_names = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP', None: '—'}

    for r, row in enumerate(desc):
        for c, tile in enumerate(row):
            y = nrows - 1 - r
            rect = patches.FancyBboxPatch(
                (c + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=TILE_COLORS[tile],
                edgecolor='#BDC3C7',
                linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(c + 0.5, y + 0.5, tile,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=TILE_TEXT_COLORS[tile])

    # Agent marker
    agent_row, agent_col = divmod(state, ncols)
    y_agent = nrows - 1 - agent_row
    circle = plt.Circle((agent_col + 0.5, y_agent + 0.5), 0.28,
                         color=AGENT_COLOR, zorder=5)
    ax.add_patch(circle)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect('equal')
    ax.axis('off')

    title = f"Step {step}  |  Action: {action_names[action]}  |  Reward: {reward:.1f}"
    if done:
        tile_here = desc[divmod(state, ncols)[0]][divmod(state, ncols)[1]]
        title += "  ✓ GOAL" if tile_here == 'G' else "  ✗ HOLE"
    ax.set_title(title, fontsize=9, pad=4)


# ── episode runner ─────────────────────────────────────────────────────────────

def run_episode(env, key, max_steps=100):
    """Run one episode with a uniform random policy. Returns trajectory list."""
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    # trajectory entry: (state, action_taken, reward, done)
    trajectory = [(int(state), None, 0.0, False)]

    for _ in range(max_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        action = int(jax.random.randint(action_key, shape=(), minval=0, maxval=4))
        obs, state, reward, done, _ = env.step(step_key, jnp.int32(state), jnp.int32(action))
        trajectory.append((int(state), action, float(reward), bool(done)))
        if bool(done):
            break

    return trajectory


def save_gif(env, trajectory, path, fps=4):
    nrows, ncols = env.nrows, env.ncols
    fig_w = max(3.5, ncols * 0.9)
    fig_h = max(3.5, nrows * 0.9) + 0.6   # +0.6 for title

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#FDFEFE')
    plt.tight_layout(pad=0.5)

    def update(i):
        state, action, reward, done = trajectory[i]
        render_frame(ax, env, state, step=i, reward=reward, done=done, action=action)

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default='4x4', choices=['4x4', '8x8'],
                        help='Map size (default: 4x4)')
    parser.add_argument('--episodes', type=int, default=6,
                        help='Number of episodes to record (default: 6)')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max steps per episode (default: 100)')
    parser.add_argument('--fps', type=int, default=4,
                        help='GIF frames per second (default: 4)')
    parser.add_argument('--no-slippery', action='store_true',
                        help='Deterministic transitions (default: slippery)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    is_slippery = not args.no_slippery
    env = FrozenLakeEnv(map_name=args.map, is_slippery=is_slippery)

    run_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..', 'runs',
        f"frozen_lake_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = os.path.normpath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Map: {args.map}  |  Slippery: {is_slippery}  |  Episodes: {args.episodes}")
    print(f"Output: {run_dir}\n")

    key = jax.random.PRNGKey(args.seed)
    outcomes = []

    for ep in range(args.episodes):
        key, ep_key = jax.random.split(key)
        traj = run_episode(env, ep_key, max_steps=args.max_steps)

        final_state, _, final_reward, _ = traj[-1]
        final_tile = env.desc[divmod(final_state, env.ncols)[0]][divmod(final_state, env.ncols)[1]]
        won = final_tile == 'G'
        outcomes.append({'steps': len(traj) - 1, 'won': won, 'reward': final_reward})

        gif_path = os.path.join(run_dir, f"episode_{ep:02d}.gif")
        save_gif(env, traj, gif_path, fps=args.fps)

        status = "GOAL" if won else "HOLE" if final_tile == 'H' else "TIMEOUT"
        print(f"  Episode {ep:2d}: {status:7s}  steps={len(traj)-1:3d}  "
              f"reward={final_reward:.1f}  → {gif_path}")

    # Summary
    win_rate = sum(o['won'] for o in outcomes) / len(outcomes)
    avg_steps = sum(o['steps'] for o in outcomes) / len(outcomes)
    summary = (
        f"Map: {args.map}\n"
        f"Slippery: {is_slippery}\n"
        f"Episodes: {args.episodes}\n"
        f"Win rate: {win_rate:.1%}\n"
        f"Avg steps: {avg_steps:.1f}\n"
    )
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(summary)

    print(f"\nWin rate: {win_rate:.1%}  |  Avg steps: {avg_steps:.1f}")
    print(f"Done. GIFs saved to {run_dir}/")


if __name__ == '__main__':
    main()
