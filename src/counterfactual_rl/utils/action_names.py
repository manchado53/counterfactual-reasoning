"""
Utility functions for translating SMAC action indices to human-readable names.
"""

from typing import Tuple, List


def get_action_name(action_id: int, n_enemies: int, map_type: str = "3m") -> str:
    """
    Convert SMAC action index to human-readable name.
    
    Args:
        action_id: The action index (0 to n_actions-1)
        n_enemies: Number of enemy units in the map
        map_type: Map type (e.g., "3m", "MMM") - affects whether unit 6+ is attack/heal
    
    Returns:
        Human-readable action name
    
    SMAC Action Space:
        0: no-op (only for dead agents)
        1: stop
        2: move north
        3: move south
        4: move east
        5: move west
        6+: attack enemy 0, attack enemy 1, ..., attack enemy N-1
            (or heal ally if MMM map and agent is medivac)
    """
    if action_id == 0:
        return "no-op"
    elif action_id == 1:
        return "stop"
    elif action_id == 2:
        return "move_north"
    elif action_id == 3:
        return "move_south"
    elif action_id == 4:
        return "move_east"
    elif action_id == 5:
        return "move_west"
    else:
        # Attack/heal actions start at index 6
        target_id = action_id - 6
        if target_id < n_enemies:
            return f"attack_enemy_{target_id}"
        else:
            return f"invalid_action_{action_id}"


def translate_joint_action(
    joint_action: Tuple[int, ...],
    n_enemies: int,
    map_type: str = "3m"
) -> List[str]:
    """
    Translate a joint action tuple to list of human-readable action names.
    
    Args:
        joint_action: Tuple of action indices, one per agent (e.g., (8, 7, 6))
        n_enemies: Number of enemy units in the map
        map_type: Map type (e.g., "3m", "MMM")
    
    Returns:
        List of action names, one per agent
    
    Example:
        >>> translate_joint_action((1, 4, 8), n_enemies=3)
        ['stop', 'move_east', 'attack_enemy_2']
    """
    return [
        get_action_name(action_id, n_enemies, map_type)
        for action_id in joint_action
    ]


def format_joint_action(
    joint_action: Tuple[int, ...],
    n_enemies: int,
    map_type: str = "3m",
    agent_prefix: str = "Agent"
) -> str:
    """
    Format a joint action as a readable string with agent labels.
    
    Args:
        joint_action: Tuple of action indices
        n_enemies: Number of enemy units
        map_type: Map type
        agent_prefix: Prefix for agent labels (default: "Agent")
    
    Returns:
        Formatted string
    
    Example:
        >>> format_joint_action((1, 4, 8), n_enemies=3)
        'Agent 0: stop | Agent 1: move_east | Agent 2: attack_enemy_2'
    """
    action_names = translate_joint_action(joint_action, n_enemies, map_type)
    
    parts = [
        f"{agent_prefix} {i}: {name}"
        for i, name in enumerate(action_names)
    ]
    
    return " | ".join(parts)
