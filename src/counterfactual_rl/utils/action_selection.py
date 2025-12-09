"""
Action selection utilities for counterfactual analysis.

Provides efficient methods for selecting actions in large action spaces.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


def beam_search_top_k_joint_actions(
    valid_actions: List[List[int]],
    action_probs: Optional[List[Dict[int, float]]] = None,
    k: int = 20
) -> List[Tuple[int, ...]]:
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
    
    Returns:
        List of top-K joint actions as tuples, ordered by probability (highest first).
        Each tuple has length n_agents.
    
    Example:
        >>> valid_actions = [[0, 1, 4], [2, 3], [1, 4, 5]]  # 3 agents
        >>> top_k = beam_search_top_k_joint_actions(valid_actions, k=10)
        >>> print(top_k[0])  # Most probable joint action, e.g., (0, 2, 1)
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
    
    return [beam[0] for beam in beams]


