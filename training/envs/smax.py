"""SMAX environment creation and interaction helpers."""

import numpy as np
from typing import Dict, List

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario


def create_smax_env(scenario: str = '3m', seed: int = 0, obs_type: str = 'world_state'):
    """
    Create SMAX environment with heuristic enemy.

    Args:
        scenario: SMAX scenario name (e.g., '3m', '8m', '3s5z')
        seed: Random seed
        obs_type: 'world_state' (72 dims for 3m) or 'concatenated' (225 dims for 3m)

    Returns:
        Tuple of (env, jax_key, env_info)
    """
    scenario_obj = map_name_to_scenario(scenario)
    env = make('HeuristicEnemySMAX', scenario=scenario_obj, won_battle_bonus=10.0)

    key = jax.random.PRNGKey(seed)

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
    """Extract global state from SMAX observations."""
    if obs_type == 'world_state':
        return np.array(obs["world_state"])
    else:
        return np.concatenate([np.array(obs[agent]) for agent in agent_names])


def get_action_masks(env, state) -> np.ndarray:
    """Get action masks for all agents from SMAX environment."""
    avail_actions = env.get_avail_actions(state)
    agent_names = list(env.agents)
    return np.array([np.array(avail_actions[agent]) for agent in agent_names])


def get_global_reward(rewards: Dict, agent_names: List[str]) -> float:
    """Get team reward (all agents receive the same reward in SMAX)."""
    return float(rewards[agent_names[0]])


def is_done(dones: Dict) -> bool:
    """Check if episode is done."""
    return bool(dones.get("__all__", False))
