"""
Action selection utilities for counterfactual analysis.

Provides efficient methods for selecting actions in large action spaces.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np


def convert_mask_to_indices(action_mask: List[List[int]]) -> List[List[int]]:
    """
    Convert binary action mask to indices of valid actions.

    Args:
        action_mask: Binary mask per agent, shape (n_agents, n_actions)
                    Example: [[0,1,1,0,1], [1,0,1,1,0], ...]
                    Where 1 = valid action, 0 = invalid action

    Returns:
        List of valid action indices per agent
        Example: [[1,2,4], [0,2,3], ...]
    """
    return [
        [i for i, valid in enumerate(agent_mask) if valid == 1]
        for agent_mask in action_mask
    ]


def beam_search_top_k_joint_actions(
    valid_actions: List[List[int]],
    action_probs: Optional[List[Dict[int, float]]] = None,
    k: int = 20,
    return_probs: bool = False
) -> Union[List[Tuple[int, ...]], Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], float]]]:
    """
    Beam search for top-K joint actions using only valid actions.

    For MultiDiscrete action spaces where the joint action space is exponentially large
    (n_actions^n_agents), this efficiently finds the top-K most probable joint actions
    without enumerating all possibilities.

    Complexity: O(n_agents × K × max_valid_actions) instead of O(n_actions^n_agents)

    Args:
        valid_actions: List of valid action indices per agent.
                      e.g., [[0, 1, 4], [2, 3], [1, 4, 5]] for 3 agents
        action_probs: Optional dict mapping action_id -> probability per agent.
                     If None, assumes uniform probability over valid actions.
                     e.g., [{0: 0.5, 1: 0.3, 4: 0.2}, {2: 0.6, 3: 0.4}, ...]
        k: Number of top joint actions to return (default: 20)
        return_probs: If True, also return normalized joint action probabilities.

    Returns:
        If return_probs=False: List of top-K joint actions as tuples.
        If return_probs=True: Tuple of (actions_list, probs_dict) where probs_dict
            maps each joint action to its normalized probability.

    Example:
        >>> valid_actions = [[0, 1, 4], [2, 3], [1, 4, 5]]  # 3 agents
        >>> top_k = beam_search_top_k_joint_actions(valid_actions, k=10)
        >>> print(top_k[0])  # Most probable joint action, e.g., (0, 2, 1)
        >>> # With probabilities:
        >>> actions, probs = beam_search_top_k_joint_actions(valid_actions, k=10, return_probs=True)
        >>> print(probs[actions[0]])  # Probability of most likely action
    """
    n_agents = len(valid_actions)
    
    if n_agents == 0:
        return []
    
    # If no probs provided, use uniform distribution
    if action_probs is None:
        action_probs = [
            {a: 1.0 / len(valid_actions[i]) if len(valid_actions[i]) > 0 else 0.0
             for a in valid_actions[i]}
            for i in range(n_agents)
        ]
    
    # Convert to log probs for numerical stability
    log_probs = [
        {a: np.log(p + 1e-10) for a, p in agent_probs.items()}
        for agent_probs in action_probs
    ]
    
    # Handle edge case: first agent has no valid actions
    if len(valid_actions[0]) == 0:
        return []
    
    # Initialize beams with agent 0's valid actions
    # Each beam is (joint_action_tuple, cumulative_log_prob)
    beams: List[Tuple[Tuple[int, ...], float]] = [
        ((action,), log_probs[0][action])
        for action in valid_actions[0]
    ]
    
    # Prune to top-K
    beams.sort(key=lambda x: x[1], reverse=True)
    beams = beams[:k]
    
    # Extend beams for each subsequent agent
    for agent_id in range(1, n_agents):
        agent_valid_actions = valid_actions[agent_id]
        agent_log_probs = log_probs[agent_id]
        
        # Handle edge case: no valid actions for this agent
        if len(agent_valid_actions) == 0:
            continue
        
        new_beams = []
        for joint_action, cum_log_prob in beams:
            for action in agent_valid_actions:
                new_joint = joint_action + (action,)
                new_log_prob = cum_log_prob + agent_log_probs[action]
                new_beams.append((new_joint, new_log_prob))
        
        # Prune to top-K
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:k]

    if return_probs:
        # Convert log probs back to probs and normalize over the top-K
        actions = [beam[0] for beam in beams]
        raw_probs = {beam[0]: np.exp(beam[1]) for beam in beams}
        total = sum(raw_probs.values())
        if total > 0:
            normalized_probs = {k: v / total for k, v in raw_probs.items()}
        else:
            # Fallback to uniform if all probs are zero
            normalized_probs = {k: 1.0 / len(actions) for k in actions}
        return actions, normalized_probs

    return [beam[0] for beam in beams]


