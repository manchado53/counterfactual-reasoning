"""Utility functions for SMAX DQN training."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple

import jax
import jax.numpy as jnp
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


def evaluate(agent, n_episodes: int = 100, seed: int = 42, parallel: bool = False):
    """
    Greedy evaluation of a trained agent.

    Runs episodes with epsilon=0, reports win rate, avg allies alive,
    avg return, and avg episode length.

    Args:
        agent: Trained DQN agent (must have .env, .env_info, .select_action(), .epsilon)
        n_episodes: Number of evaluation episodes
        seed: Random seed
        parallel: If True, use vectorized JAX evaluation (requires agent.make_policy_fn())

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

    # Save and disable exploration
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
            action_dict = {agent_name: joint_action[i] for i, agent_name in enumerate(agent_names)}

            obs, state, rewards, dones, _ = env.step(step_key, state, action_dict)

            episode_return += get_global_reward(rewards, agent_names)
            episode_length += 1
            done = is_done(dones)

        # Check win condition from final state
        allies_alive = int(np.sum(np.array(state.unit_alive[:num_allies])))
        enemies_alive = int(np.sum(np.array(state.unit_alive[num_allies:])))
        won = (enemies_alive == 0) and (allies_alive > 0)

        wins += int(won)
        total_allies_alive += allies_alive
        total_return += episode_return
        total_length += episode_length

    # Restore epsilon
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
    """
    Vectorized greedy evaluation using jax.vmap — JAX agents only.

    Runs all episodes in parallel using jax.lax.scan over timesteps
    and jax.vmap over episodes.
    """
    env = agent.env
    agent_names = agent.env_info['agent_names']
    num_allies = env.num_allies
    max_steps = 100  # SMAX default max steps

    # Get JIT-compatible policy function
    policy_fn = agent.make_policy_fn()

    # Generate keys for all episodes
    master_key = jax.random.PRNGKey(seed)
    episode_keys = jax.random.split(master_key, n_episodes)

    def run_single_episode(episode_key):
        """Run one episode with greedy policy, return (return, length, allies_alive, won)."""
        reset_key, run_key = jax.random.split(episode_key)
        obs, state = env.reset(reset_key)

        def scan_step(carry, _):
            state_c, obs_c, key_c, cum_return, length, done_flag = carry

            key_c, policy_key, step_key = jax.random.split(key_c, 3)

            # Get available actions and policy action
            avail_actions = env.get_avail_actions(state_c)
            action_dict = policy_fn(policy_key, obs_c, avail_actions)

            # Step environment
            new_obs, new_state, rewards, dones, _ = env.step(step_key, state_c, action_dict)

            # Accumulate reward (masked by done)
            step_reward = jnp.float32(0.0)
            for agent_name in agent_names:
                step_reward = step_reward + rewards[agent_name]

            cum_return = cum_return + step_reward * (1.0 - done_flag)
            length = length + (1.0 - done_flag)
            done_flag = jnp.where(dones["__all__"], 1.0, done_flag)

            return (new_state, new_obs, key_c, cum_return, length, done_flag), None

        init_carry = (state, obs, run_key, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
        (final_state, _, _, episode_return, episode_length, _), _ = jax.lax.scan(
            scan_step, init_carry, None, length=max_steps
        )

        # Check win condition (HeuristicEnemySMAX wraps SMAX State inside .state)
        smax_state = final_state.state
        allies_alive = jnp.sum(smax_state.unit_alive[:num_allies])
        enemies_alive = jnp.sum(smax_state.unit_alive[num_allies:])
        won = (enemies_alive == 0) & (allies_alive > 0)

        return episode_return, episode_length, allies_alive, won.astype(jnp.float32)

    # Vmap over all episodes and JIT compile
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
