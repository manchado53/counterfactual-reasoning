"""
Environment state management for counterfactual analysis
"""

import copy
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym

from counterfactual_rl.environments.base import StateManager, EnvironmentConfig


class FrozenLakeStateManager(StateManager):
    """
    Manages state cloning and restoration for FrozenLake environment.

    FrozenLake state consists of:
    - Current position (s)
    - RNG state (for stochastic transitions when slippery=True)

    This enables counterfactual "what-if" analysis by allowing us to
    reset the environment to an exact previous state and test alternative actions.
    """

    @staticmethod
    def clone_state(env: gym.Env) -> Dict[str, Any]:
        """
        Clone the complete state of a FrozenLake environment.

        Args:
            env: FrozenLake environment instance

        Returns:
            Dictionary containing all state information needed for restoration
        """
        # Get the unwrapped environment to access internal state
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env

        state_dict = {
            's': unwrapped.s,  # Current state (position)
            'np_random_state': copy.deepcopy(unwrapped.np_random.bit_generator.state),
        }

        # IMPORTANT: Save TimeLimit wrapper's elapsed_steps to prevent step budget exhaustion
        # During counterfactual rollouts, many steps are taken which can exhaust the episode
        # step limit. We need to restore this counter after rollouts.
        if hasattr(env, '_elapsed_steps'):
            state_dict['_elapsed_steps'] = env._elapsed_steps

        return state_dict

    @staticmethod
    def restore_state(env: gym.Env, state_dict: Dict[str, Any]) -> None:
        """
        Restore a FrozenLake environment to a previously cloned state.

        This allows us to "rewind time" and test alternative actions from
        the exact same state, including the same RNG state for stochastic
        environments.

        Args:
            env: FrozenLake environment instance
            state_dict: State dictionary from clone_state()
        """
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env

        # Restore position
        unwrapped.s = state_dict['s']

        # Restore RNG state
        unwrapped.np_random.bit_generator.state = copy.deepcopy(state_dict['np_random_state'])

        # Restore TimeLimit wrapper's step counter
        # This is CRITICAL to prevent premature episode termination after counterfactual rollouts
        if '_elapsed_steps' in state_dict and hasattr(env, '_elapsed_steps'):
            env._elapsed_steps = state_dict['_elapsed_steps']

    @staticmethod
    def get_state_info(env: gym.Env) -> Dict[str, Any]:
        """Get human-readable information about current state.
        
        Args:
            env: FrozenLake environment instance
            
        Returns:
            Dictionary with state and position information
        """
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        state = unwrapped.s
        row = state // 4
        col = state % 4
        
        return {
            'state': state,
            'position': (row, col),
            'row': row,
            'col': col,
        }
    
    @staticmethod
    def get_grid_position(env: gym.Env) -> Tuple[int, int]:
        """Get (row, col) grid position for FrozenLake (4x4 grid).
        
        Args:
            env: FrozenLake environment instance
            
        Returns:
            (row, col) tuple
        """
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        state = unwrapped.s
        return (state // 4, state % 4)
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """FrozenLake is a 4x4 grid."""
        return (4, 4)


class FrozenLakeConfig(EnvironmentConfig):
    """Configuration for FrozenLake environments."""
    
    def __init__(self, slippery: bool = False):
        self.slippery = slippery
    
    @property
    def observation_space_size(self) -> int:
        """FrozenLake has 16 states (4x4 grid)."""
        return 16
    
    @property
    def action_space_size(self) -> int:
        """FrozenLake has 4 discrete actions (LEFT, DOWN, RIGHT, UP)."""
        return 4
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """FrozenLake is a 4x4 grid."""
        return (4, 4)
    
    @property
    def state_manager_class(self):
        """Returns FrozenLakeStateManager class."""
        return FrozenLakeStateManager


# NOTE: StateManagerFactory has been removed in favor of the registry pattern.
# Use counterfactual_rl.environments.registry instead:
#
#   from counterfactual_rl.environments import registry
#   state_manager = registry.get_state_manager("FrozenLake-v1")
#   config = registry.get_config("FrozenLake-v1")
