"""Registry system for environment-specific visualizers.

This module implements a registry pattern for managing environment visualizers.
Each environment can have its own custom visualizer registered.

Quick Start:
    from counterfactual_rl.visualization import registry
    
    # Register a visualizer
    registry.register("FrozenLake-v1", FrozenLakeVisualizer)
    
    # Get a visualizer
    visualizer = registry.get_visualizer("FrozenLake-v1")
    
    # List all registered visualizers
    print(registry.list_registered())
"""

# Module-level registry for visualizers
_VISUALIZERS = {}


def register(env_id, VisualizerClass):
    """Register a visualizer for an environment.
    
    Args:
        env_id (str): Environment identifier (e.g., "FrozenLake-v1")
        VisualizerClass: The visualizer class to register
        
    Raises:
        ValueError: If environment already registered
        
    Example:
        >>> from counterfactual_rl.visualization import registry
        >>> class MyVisualizer:
        ...     pass
        >>> registry.register("MyEnv-v1", MyVisualizer)
    """
    if env_id in _VISUALIZERS:
        raise ValueError(
            f"Visualizer for '{env_id}' already registered. "
            f"Use a different environment ID or unregister first."
        )
    _VISUALIZERS[env_id] = VisualizerClass


def get_visualizer(env_id, *args, **kwargs):
    """Get an instantiated visualizer for an environment.
    
    Args:
        env_id (str): Environment identifier (e.g., "FrozenLake-v1")
        *args: Positional arguments to pass to visualizer constructor
        **kwargs: Keyword arguments to pass to visualizer constructor
        
    Returns:
        An instance of the registered visualizer
        
    Raises:
        KeyError: If environment not registered
        
    Example:
        >>> from counterfactual_rl.visualization import registry
        >>> visualizer = registry.get_visualizer("FrozenLake-v1")
        >>> visualizer.plot_grid()
    """
    if env_id not in _VISUALIZERS:
        available = list(_VISUALIZERS.keys())
        raise KeyError(
            f"No visualizer registered for '{env_id}'. "
            f"Available environments: {available}"
        )
    VisualizerClass = _VISUALIZERS[env_id]
    return VisualizerClass(*args, **kwargs)


def list_registered():
    """List all registered visualizers.
    
    Returns:
        list: List of registered environment IDs
        
    Example:
        >>> from counterfactual_rl.visualization import registry
        >>> print(registry.list_registered())
        ['FrozenLake-v1', 'FrozenLake8x8-v1', 'Taxi-v3']
    """
    return list(_VISUALIZERS.keys())


def is_registered(env_id):
    """Check if a visualizer is registered for an environment.
    
    Args:
        env_id (str): Environment identifier (e.g., "FrozenLake-v1")
        
    Returns:
        bool: True if registered, False otherwise
        
    Example:
        >>> from counterfactual_rl.visualization import registry
        >>> registry.is_registered("FrozenLake-v1")
        True
    """
    return env_id in _VISUALIZERS
