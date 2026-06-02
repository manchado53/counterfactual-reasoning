"""
Generate a GIF of 10 episodes with the trained FrozenLake policy.
Run: python -m counterfactual_rl.analysis.claim1.frozen_lake.make_gif
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import imageio.v2 as imageio
import io

from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv
from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN

MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
N = 8
HOLE_STATES = {r * N + c for r, row in enumerate(MAP) for c, ch in enumerate(row) if ch == 'H'}
GOAL_STATE  = next(r * N + c for r, row in enumerate(MAP) for c, ch in enumerate(row) if ch == 'G')

ACTION_ARROWS = {0: '←', 1: '↓', 2: '→', 3: '↑'}
CELL_COLORS = {
    'F': '#E3F2FD',
    'S': '#C8E6C9',
    'H': '#212121',
    'G': '#FFF9C4',
}


def render_frame(state, episode, step, result=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.axis('off')

    for r, row in enumerate(MAP):
        for c, ch in enumerate(row):
            color = CELL_COLORS.get(ch, '#FFFFFF')
            rect = patches.Rectangle((c, N - 1 - r), 1, 1,
                                      linewidth=0.5, edgecolor='#90A4AE',
                                      facecolor=color)
            ax.add_patch(rect)
            if ch == 'H':
                ax.text(c + 0.5, N - 1 - r + 0.5, '✕', ha='center', va='center',
                        fontsize=14, color='white', fontweight='bold')
            elif ch == 'G':
                ax.text(c + 0.5, N - 1 - r + 0.5, 'G', ha='center', va='center',
                        fontsize=13, color='#F57F17', fontweight='bold')

    # Agent
    if result is None:
        ar, ac = divmod(int(state), N)
        circle = plt.Circle((ac + 0.5, N - 1 - ar + 0.5), 0.3,
                             color='#1565C0', zorder=5)
        ax.add_patch(circle)
    elif result == 'win':
        ar, ac = divmod(GOAL_STATE, N)
        ax.text(ac + 0.5, N - 1 - ar + 0.5, '★', ha='center', va='center',
                fontsize=20, color='#F9A825', zorder=6)
    elif result == 'hole':
        ar, ac = divmod(int(state), N)
        ax.text(ac + 0.5, N - 1 - ar + 0.5, '💀', ha='center', va='center',
                fontsize=16, zorder=6)

    title = f'Episode {episode+1}  |  Step {step}'
    if result == 'win':
        title += '  ✓ WIN'
    elif result == 'hole':
        title += '  ✗ HOLE'
    ax.set_title(title, fontsize=11, pad=6)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def run_episode(env, params, network, seed):
    rng = jax.random.PRNGKey(seed)
    s = jnp.int32(0)
    frames = []
    frames.append(render_frame(s, seed, 0))

    for step in range(1, 1000):
        a = jnp.argmax(network.apply(params, s))
        rng, sk = jax.random.split(rng)
        _, ns, r, done, _ = env.step(sk, s, a)

        if done:
            result = 'win' if float(r) > 0 else 'hole'
            frames.append(render_frame(ns, seed, step, result=result))
            # Hold final frame
            for _ in range(8):
                frames.append(frames[-1])
            return frames, step, result

        s = ns
        frames.append(render_frame(s, seed, step))

    return frames, 1000, 'timeout'


def main():
    env = FrozenLakeEnv(map_name='8x8', is_slippery=True)
    agent = FrozenLakeDQN()
    ckpt = Path(__file__).parent / 'checkpoints' / 'seed_0' / 'trained.pkl'
    agent.load(ckpt)

    out_dir = Path(__file__).parents[5] / 'docs' / 'figures' / 'claim1'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for ep in range(10):
        print(f'  Episode {ep+1}/10 ...', end=' ', flush=True)
        frames, length, result = run_episode(env, agent.params, agent.network, seed=ep)
        print(f'{length} steps  ({result})')
        all_frames.extend(frames)
        # Brief black separator between episodes
        sep = np.zeros_like(frames[0])
        for _ in range(3):
            all_frames.append(sep)

    out_path = out_dir / 'trained_policy_episodes.gif'
    imageio.mimwrite(str(out_path), all_frames, fps=8, loop=0)
    print(f'\nGIF saved → {out_path}  ({len(all_frames)} frames)')


if __name__ == '__main__':
    main()
