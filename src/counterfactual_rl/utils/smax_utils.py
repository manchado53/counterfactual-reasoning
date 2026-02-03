"""
SMAX (JaxMARL) utility functions for counterfactual analysis.

Provides utilities for:
- Action format conversion (tuple <-> dict)
- Available actions mask extraction
- Agent name management
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def get_agent_names(env) -> List[str]:
    """
    Get ordered list of agent names from SMAX environment.

    Args:
        env: SMAX environment

    Returns:
        List of agent names, e.g., ["ally_0", "ally_1", "ally_2"]
    """
    return list(env.agents)


def tuple_to_action_dict(
    action_tuple: Tuple[int, ...],
    agent_names: List[str]
) -> Dict[str, int]:
    """
    Convert tuple of actions to SMAX action dictionary.

    Args:
        action_tuple: Joint action as tuple, e.g., (4, 1, 2)
        agent_names: List of agent names in order, e.g., ["ally_0", "ally_1", "ally_2"]

    Returns:
        Action dict, e.g., {"ally_0": 4, "ally_1": 1, "ally_2": 2}

    Raises:
        ValueError: If action tuple length doesn't match agent names length
    """
    if len(action_tuple) != len(agent_names):
        raise ValueError(
            f"Action tuple length {len(action_tuple)} != agent names length {len(agent_names)}"
        )
    return {agent: int(action) for agent, action in zip(agent_names, action_tuple)}


def action_dict_to_tuple(
    action_dict: Dict[str, Any],
    agent_names: List[str]
) -> Tuple[int, ...]:
    """
    Convert SMAX action dictionary to tuple.

    Args:
        action_dict: Action dict, e.g., {"ally_0": 4, "ally_1": 1, "ally_2": 2}
        agent_names: List of agent names in order (defines tuple ordering)

    Returns:
        Joint action as tuple, e.g., (4, 1, 2)
    """
    return tuple(int(action_dict[agent]) for agent in agent_names)


def get_valid_actions_mask(env, state) -> List[List[int]]:
    """
    Get available actions mask for all agents from SMAX environment.

    Args:
        env: SMAX environment
        state: Current JAX state

    Returns:
        List of binary masks per agent, e.g., [[1,1,0,1], [1,0,1,1], ...]
        where 1 = valid action, 0 = invalid action
    """
    avail_actions = env.get_avail_actions(state)  # Dict[str, Array]

    # Convert to list of lists, maintaining agent order
    agent_names = get_agent_names(env)
    masks = []
    for agent in agent_names:
        mask = avail_actions[agent]
        # Convert JAX array to Python list of ints
        masks.append([int(x) for x in np.array(mask)])

    return masks


def flatten_observations(
    obs: Dict[str, Any],
    agent_names: List[str]
) -> np.ndarray:
    """
    Flatten observation dict to single array for policy input.

    Args:
        obs: Dict mapping agent names to observation arrays
        agent_names: List of agent names (defines ordering)

    Returns:
        Concatenated numpy array of all observations
    """
    return np.concatenate([np.array(obs[agent]) for agent in agent_names])


def sum_rewards(rewards: Dict[str, Any], agent_names: List[str]) -> float:
    """
    Sum rewards across all agents for team reward.

    Args:
        rewards: Dict mapping agent names to individual rewards
        agent_names: List of agent names

    Returns:
        Total team reward
    """
    return sum(float(rewards[agent]) for agent in agent_names)
