"""Utility functions for SMAX DQN training."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

import jax
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario


def create_smax_env(scenario: str = '3m', seed: int = 0, obs_type: str = 'world_state'):
    """
    Create SMAX environment with heuristic enemy.

    Uses HeuristicEnemySMAX which provides a scripted opponent,
    matching the existing codebase pattern.

    Args:
        scenario: SMAX scenario name (e.g., '3m', '8m', '3s5z')
            Common scenarios: '3m', '8m', '2s3z', '3s5z', '5m_vs_6m'
        seed: Random seed
        obs_type: 'world_state' (72 dims for 3m) or 'concatenated' (225 dims for 3m)

    Returns:
        Tuple of (env, jax_key, env_info)

    Example:
        env, key, env_info = create_smax_env(scenario='3m', obs_type='world_state')
        obs, state = env.reset(key)
        global_state = get_global_state(obs, env_info['agent_names'], env_info['obs_type'])
        action_masks = get_action_masks(env, state)
    """
    scenario_obj = map_name_to_scenario(scenario)
    env = make('HeuristicEnemySMAX', scenario=scenario_obj)

    key = jax.random.PRNGKey(seed)

    # Get environment dimensions
    obs, state = env.reset(key)
    agent_names = list(env.agents)

    single_obs_dim = obs[agent_names[0]].shape[0]
    if obs_type == 'world_state':
        obs_dim = obs["world_state"].shape[0]
    else:
        obs_dim = single_obs_dim * len(agent_names)
    num_agents = len(agent_names)
    actions_per_agent = env.action_space(agent_names[0]).n

    env_info = {
        'obs_dim': obs_dim,
        'single_obs_dim': single_obs_dim,
        'num_agents': num_agents,
        'actions_per_agent': actions_per_agent,
        'scenario': scenario,
        'agent_names': agent_names,
        'obs_type': obs_type,
    }

    return env, key, env_info


def get_global_state(obs: Dict, agent_names: List[str], obs_type: str = 'world_state') -> np.ndarray:
    """
    Extract global state from SMAX observations.

    Args:
        obs: Dict mapping agent names to observation arrays
        agent_names: Ordered list of agent names
        obs_type: 'world_state' or 'concatenated'

    Returns:
        Global state array as numpy
    """
    if obs_type == 'world_state':
        return np.array(obs["world_state"])
    else:
        return np.concatenate([np.array(obs[agent]) for agent in agent_names])


def get_action_masks(env, state) -> np.ndarray:
    """
    Get action masks for all agents from SMAX environment.

    Args:
        env: Raw SMAX environment
        state: Current environment state

    Returns:
        Action masks array of shape (num_agents, actions_per_agent)
    """
    avail_actions = env.get_avail_actions(state)
    agent_names = list(env.agents)

    masks = []
    for agent in agent_names:
        mask = np.array(avail_actions[agent])
        masks.append(mask)

    return np.array(masks)


def get_global_reward(rewards: Dict, agent_names: List[str]) -> float:
    """
    Sum rewards across all agents.

    Args:
        rewards: Dict mapping agent names to rewards
        agent_names: List of agent names

    Returns:
        Total team reward
    """
    return sum(float(rewards[agent]) for agent in agent_names)


def is_done(dones: Dict) -> bool:
    """
    Check if episode is done.

    Args:
        dones: Dict mapping agent names to done flags

    Returns:
        True if episode is finished
    """
    return bool(dones.get("__all__", False))


def record_episode(agent, env=None, seed: int = 42, greedy: bool = True):
    """
    Run an episode with the agent and record state sequence for visualization.

    Args:
        agent: Trained DQN agent
        env: Optional environment (uses agent's env if None)
        seed: Random seed for episode
        greedy: If True, disable exploration (epsilon=0)

    Returns:
        Tuple of (state_seq, episode_return, episode_length)
        - state_seq: List of (key, state, action_dict) tuples for SMAXVisualizer
        - episode_return: Total return for the episode
        - episode_length: Number of steps
    """
    if env is None:
        env = agent.env

    agent_names = agent.env_info['agent_names']

    # Save and optionally disable exploration
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
        action_dict = {agent_name: joint_action[i] for i, agent_name in enumerate(agent_names)}

        # Store for visualization (key, state, actions)
        state_seq.append((step_key, state, action_dict))

        obs, state, rewards, dones, _ = env.step(step_key, state, action_dict)

        episode_return += get_global_reward(rewards, agent_names)
        episode_length += 1
        done = is_done(dones)

    # Restore epsilon
    agent.epsilon = original_epsilon

    return state_seq, episode_return, episode_length


def save_gameplay_gif(env, state_seq, save_path: str = "gameplay.gif"):
    """
    Save recorded episode as GIF using JaxMARL's SMAXVisualizer.

    Args:
        env: SMAX environment
        state_seq: List of (key, state, action_dict) tuples from record_episode()
        save_path: Path to save the GIF

    Note:
        Requires ffmpeg or pillow for video/GIF creation.
    """
    import subprocess
    import shutil

    # Check for ffmpeg
    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print("Warning: ffmpeg not found. Trying with pillow (may be slower)...")

    from jaxmarl.viz.visualizer import SMAXVisualizer
    import matplotlib.animation as animation

    # Set writer based on availability
    if has_ffmpeg:
        writer = 'ffmpeg'
    else:
        writer = 'pillow'

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
    """
    Plot training curves.

    Args:
        episode_returns: List of episode returns
        episode_lengths: List of episode lengths
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Episode returns
    axes[0].plot(episode_returns, alpha=0.3, label='Episode Return')
    if len(episode_returns) >= 100:
        smoothed = np.convolve(episode_returns, np.ones(100)/100, mode='valid')
        axes[0].plot(range(99, len(episode_returns)), smoothed, label='100-ep Average')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('Training Returns')
    axes[0].legend()

    # Episode lengths
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
