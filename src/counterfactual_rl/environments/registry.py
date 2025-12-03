"""
Simple environment registry - replaces factory pattern.

Maps environment IDs to StateManager and EnvironmentConfig classes.
"""

from typing import Dict, Type, Optional
from counterfactual_rl.environments.base import StateManager, EnvironmentConfig


# Registry dictionaries
_STATE_MANAGERS: Dict[str, Type[StateManager]] = {}
_CONFIGS: Dict[str, Type[EnvironmentConfig]] = {}


def register(
    env_id: str,
    state_manager_class: Type[StateManager],
    config_class: Type[EnvironmentConfig],
) -> None:
    """Register an environment.
    
    Args:
        env_id: Environment ID (e.g., 'FrozenLake-v1')
        state_manager_class: StateManager implementation
        config_class: EnvironmentConfig implementation
    
    Example:
        >>> register('FrozenLake-v1', FrozenLakeStateManager, FrozenLakeConfig)
    """
    _STATE_MANAGERS[env_id] = state_manager_class
    _CONFIGS[env_id] = config_class


def get_state_manager(env_id: str) -> StateManager:
    """Get state manager instance for an environment.
    
    Args:
        env_id: Environment ID
        
    Returns:
        Instantiated StateManager
        
    Raises:
        KeyError: If environment not registered
        
    Example:
        >>> sm = get_state_manager('FrozenLake-v1')
    """
    if env_id not in _STATE_MANAGERS:
        available = list(_STATE_MANAGERS.keys())
        raise KeyError(
            f"Environment '{env_id}' not registered. "
            f"Available: {available}"
        )
    return _STATE_MANAGERS[env_id]()


def get_config(env_id: str, **kwargs) -> EnvironmentConfig:
    """Get config instance for an environment.
    
    Args:
        env_id: Environment ID
        **kwargs: Arguments to pass to config constructor
        
    Returns:
        Instantiated EnvironmentConfig
        
    Raises:
        KeyError: If environment not registered
        
    Example:
        >>> config = get_config('FrozenLake-v1', slippery=False)
    """
    if env_id not in _CONFIGS:
        available = list(_CONFIGS.keys())
        raise KeyError(
            f"Environment '{env_id}' not registered. "
            f"Available: {available}"
        )
    return _CONFIGS[env_id](**kwargs)


def list_registered() -> list:
    """List all registered environment IDs.
    
    Returns:
        List of registered environment IDs
        
    Example:
        >>> envs = list_registered()
        >>> print(envs)
        ['FrozenLake-v1', 'FrozenLake8x8-v1']
    """
    return list(_STATE_MANAGERS.keys())


def is_registered(env_id: str) -> bool:
    """Check if an environment is registered.
    
    Args:
        env_id: Environment ID
        
    Returns:
        True if registered, False otherwise
        
    Example:
        >>> is_registered('FrozenLake-v1')
        True
    """
    return env_id in _STATE_MANAGERS
