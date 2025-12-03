"""
Abstract base classes for environment management and configuration.

This module provides interfaces that environment-specific implementations
should follow, enabling easy extension to new environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional


class StateManager(ABC):
    """Abstract base class for environment state management.
    
    Handles environment-specific logic for:
    - Capturing/restoring state
    - Converting state to human-readable format
    - Grid position extraction (if applicable)
    """
    
    @staticmethod
    @abstractmethod
    def clone_state(env) -> Dict[str, Any]:
        """Capture environment state for later restoration.
        
        Args:
            env: Gymnasium environment instance
            
        Returns:
            Dictionary containing all state information needed to restore env
        """
        pass
    
    @staticmethod
    @abstractmethod
    def restore_state(env, state_dict: Dict[str, Any]) -> None:
        """Restore environment to a previously captured state.
        
        Args:
            env: Gymnasium environment instance
            state_dict: State dictionary from clone_state()
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_state_info(env) -> Dict[str, Any]:
        """Get human-readable information about current state.
        
        Args:
            env: Gymnasium environment instance
            
        Returns:
            Dictionary with keys like 'state', 'position', etc.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_grid_position(env) -> Tuple[int, int]:
        """Get (row, col) grid position if environment is grid-based.
        
        Args:
            env: Gymnasium environment instance
            
        Returns:
            (row, col) tuple for grid environments
            
        Raises:
            NotImplementedError: If environment is not grid-based
        """
        pass
    
    @property
    @abstractmethod
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """Grid shape (rows, cols) if environment is grid-based, else None."""
        pass


class EnvironmentConfig(ABC):
    """Abstract configuration class for environments.
    
    Stores environment metadata needed by trainers and analyzers.
    """
    
    @property
    @abstractmethod
    def observation_space_size(self) -> int:
        """Size of observation space (for network architecture)."""
        pass
    
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of discrete actions available."""
        pass
    
    @property
    @abstractmethod
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """(rows, cols) if grid-based, None otherwise."""
        pass
    
    @property
    @abstractmethod
    def state_manager_class(self):
        """Returns the StateManager class for this environment."""
        pass
    
    def get_state_manager(self):
        """Instantiate and return a StateManager for this environment."""
        return self.state_manager_class()
