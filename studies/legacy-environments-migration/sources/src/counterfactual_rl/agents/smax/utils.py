"""Training helpers for SMAX DQN (evaluation, recording, plotting)."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

import jax
import jax.numpy as jnp

from counterfactual_rl.envs.smax import (
    get_global_state,
    get_action_masks,
    get_global_reward,
    is_done,
)


def record_episode(agent, env=None, seed: int = 42, greedy: bool = True):
    """
    Run an episode with the agent and record state sequence for visualization.

    Returns:
        Tuple of (state_seq, episode_return, episode_length)
        - state_seq: List of (key, state, action_dict) tuples for SMAXVisualizer
    """
    if env is None:
        env = agent.env

    agent_names = agent.env_info['agent_names']

    original_epsilon = agent.epsilon
    if greedy:
        agent.epsilon = 0.0

    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)

    state_seq = []
    episode_return = 0.0
    episode_length = 0
    done = False

    while not done:
        key, step_key = jax.random.split(key)

        global_state = get_global_state(obs, agent_names, agent.env_info['obs_type'])
        action_masks = get_action_masks(env, state)

        joint_action = agent.select_action(global_state, action_masks)
        action_dict = {name: joint_action[i] for i, name in enumerate(agent_names)}

        state_seq.append((step_key, state, action_dict))

        obs, state, rewards, dones, _ = env.step(step_key, state, action_dict)

        episode_return += get_global_reward(rewards, agent_names)
        episode_length += 1
        done = is_done(dones)

    agent.epsilon = original_epsilon

    return state_seq, episode_return, episode_length


def save_gameplay_gif(env, state_seq, save_path: str = "gameplay.gif"):
    """Save recorded episode as GIF using JaxMARL's SMAXVisualizer."""
    import shutil
    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print("Warning: ffmpeg not found. Trying with pillow...")

    from jaxmarl.viz.visualizer import SMAXVisualizer

    try:
        viz = SMAXVisualizer(env, state_seq=state_seq)
        viz.animate(view=False, save_fname=save_path)
        print(f"Saved gameplay to {save_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Try installing ffmpeg: conda install ffmpeg")


def plot_training_curves(
    episode_returns: List[float],
    episode_lengths: List[float],
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot training curves (returns and episode lengths)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(episode_returns, alpha=0.3, label='Episode Return')
    if len(episode_returns) >= 100:
        smoothed = np.convolve(episode_returns, np.ones(100)/100, mode='valid')
        axes[0].plot(range(99, len(episode_returns)), smoothed, label='100-ep Average')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('Training Returns')
    axes[0].legend()

    axes[1].plot(episode_lengths, alpha=0.3, label='Episode Length')
    if len(episode_lengths) >= 100:
        smoothed = np.convolve(episode_lengths, np.ones(100)/100, mode='valid')
        axes[1].plot(range(99, len(episode_lengths)), smoothed, label='100-ep Average')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length')
    axes[1].set_title('Episode Lengths')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    if show:
        plt.show()
    return fig


def evaluate(agent, n_episodes: int = 100, seed: int = 42, parallel: bool = False):
    """
    Greedy evaluation of a trained agent.

    Returns:
        Dict with 'win_rate', 'avg_allies_alive', 'avg_return', 'avg_length'
    """
    if parallel:
        return _evaluate_vectorized(agent, n_episodes, seed)
    else:
        return _evaluate_sequential(agent, n_episodes, seed)


def _evaluate_sequential(agent, n_episodes: int, seed: int) -> Dict:
    """Sequential greedy evaluation — works with any backend."""
    env = agent.env
    agent_names = agent.env_info['agent_names']
    num_allies = env.num_allies

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins = 0
    total_allies_alive = 0
    total_return = 0.0
    total_length = 0

    key = jax.random.PRNGKey(seed)

    for ep in range(n_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)

        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            key, step_key = jax.random.split(key)

            global_state = get_global_state(obs, agent_names, agent.env_info['obs_type'])
            action_masks = get_action_masks(env, state)

            joint_action = agent.select_action(global_state, action_masks)
            action_dict = {name: joint_action[i] for i, name in enumerate(agent_names)}

            obs, state, rewards, dones, _ = env.step_env(step_key, state, action_dict)

            episode_return += get_global_reward(rewards, agent_names)
            episode_length += 1
            done = is_done(dones)

        smax_state = state.state
        allies_alive = int(np.sum(np.array(smax_state.unit_alive[:num_allies])))
        enemies_alive = int(np.sum(np.array(smax_state.unit_alive[num_allies:])))
        won = (enemies_alive == 0) and (allies_alive > 0)

        wins += int(won)
        total_allies_alive += allies_alive
        total_return += episode_return
        total_length += episode_length

    agent.epsilon = original_epsilon

    metrics = {
        'win_rate': wins / n_episodes,
        'avg_allies_alive': total_allies_alive / n_episodes,
        'avg_return': total_return / n_episodes,
        'avg_length': total_length / n_episodes,
    }

    print(f"\nEvaluation ({n_episodes} episodes, sequential):")
    print(f"  Win rate:         {metrics['win_rate']:.1%}")
    print(f"  Avg allies alive: {metrics['avg_allies_alive']:.2f}")
    print(f"  Avg return:       {metrics['avg_return']:.2f}")
    print(f"  Avg length:       {metrics['avg_length']:.1f}")

    return metrics


def _evaluate_vectorized(agent, n_episodes: int, seed: int) -> Dict:
    """Vectorized greedy evaluation using jax.vmap — JAX agents only."""
    env = agent.env
    agent_names = agent.env_info['agent_names']
    num_allies = env.num_allies
    max_steps = 100

    policy_fn = agent.make_policy_fn()

    master_key = jax.random.PRNGKey(seed)
    episode_keys = jax.random.split(master_key, n_episodes)

    def run_single_episode(episode_key):
        reset_key, run_key = jax.random.split(episode_key)
        obs, state = env.reset(reset_key)

        def scan_step(carry, _):
            state_c, obs_c, key_c, cum_return, length, done_flag = carry

            key_c, policy_key, step_key = jax.random.split(key_c, 3)

            avail_actions = env.get_avail_actions(state_c)
            action_dict = policy_fn(policy_key, obs_c, avail_actions)

            new_obs, new_state, rewards, dones, _ = env.step_env(step_key, state_c, action_dict)

            step_reward = rewards[agent_names[0]]
            cum_return = cum_return + step_reward * (1.0 - done_flag)
            length = length + (1.0 - done_flag)

            frozen_state = jax.tree.map(
                lambda old, new: jnp.where(done_flag, old, new),
                state_c, new_state,
            )
            frozen_obs = jax.tree.map(
                lambda old, new: jnp.where(done_flag, old, new),
                obs_c, new_obs,
            )

            done_flag = jnp.where(dones["__all__"], 1.0, done_flag)

            return (frozen_state, frozen_obs, key_c, cum_return, length, done_flag), None

        init_carry = (state, obs, run_key, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
        (final_state, _, _, episode_return, episode_length, _), _ = jax.lax.scan(
            scan_step, init_carry, None, length=max_steps
        )

        smax_state = final_state.state
        allies_alive = jnp.sum(smax_state.unit_alive[:num_allies])
        enemies_alive = jnp.sum(smax_state.unit_alive[num_allies:])
        won = (enemies_alive == 0) & (allies_alive > 0)

        return episode_return, episode_length, allies_alive, won.astype(jnp.float32)

    batched_eval = jax.jit(jax.vmap(run_single_episode))
    returns, lengths, allies, wins = batched_eval(episode_keys)

    metrics = {
        'win_rate': float(jnp.mean(wins)),
        'avg_allies_alive': float(jnp.mean(allies)),
        'avg_return': float(jnp.mean(returns)),
        'avg_length': float(jnp.mean(lengths)),
    }

    print(f"\nEvaluation ({n_episodes} episodes, vectorized):")
    print(f"  Win rate:         {metrics['win_rate']:.1%}")
    print(f"  Avg allies alive: {metrics['avg_allies_alive']:.2f}")
    print(f"  Avg return:       {metrics['avg_return']:.2f}")
    print(f"  Avg length:       {metrics['avg_length']:.1f}")

    return metrics
