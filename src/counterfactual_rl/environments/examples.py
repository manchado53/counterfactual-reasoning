"""
Example implementations of StateManager for different environments.

This module shows how to extend the framework to support new environments.
Copy and adapt these examples when adding support for Taxi-v3, CartPole, etc.
"""

from typing import Dict, Any, Tuple, Optional
import copy
import gymnasium as gym

from counterfactual_rl.environments.base import StateManager, EnvironmentConfig


class TaxiStateManager(StateManager):
    """Example: State manager for Taxi-v3 environment (5x5 grid).
    
    NOT IMPLEMENTED - This is a template for future work.
    """
    
    @staticmethod
    def clone_state(env: gym.Env) -> Dict[str, Any]:
        """Clone state of Taxi environment."""
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        state_dict = {
            's': unwrapped.s,  # Current state
            'np_random_state': copy.deepcopy(unwrapped.np_random.bit_generator.state),
        }
        
        if hasattr(env, '_elapsed_steps'):
            state_dict['_elapsed_steps'] = env._elapsed_steps
        
        return state_dict
    
    @staticmethod
    def restore_state(env: gym.Env, state_dict: Dict[str, Any]) -> None:
        """Restore state of Taxi environment."""
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        unwrapped.s = state_dict['s']
        unwrapped.np_random.bit_generator.state = copy.deepcopy(state_dict['np_random_state'])
        
        if '_elapsed_steps' in state_dict and hasattr(env, '_elapsed_steps'):
            env._elapsed_steps = state_dict['_elapsed_steps']
    
    @staticmethod
    def get_state_info(env: gym.Env) -> Dict[str, Any]:
        """Get state info for Taxi (5x5 grid)."""
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        state = unwrapped.s
        row = state // 25
        col = state % 25
        
        return {
            'state': state,
            'position': (row, col),
            'row': row,
            'col': col,
        }
    
    @staticmethod
    def get_grid_position(env: gym.Env) -> Tuple[int, int]:
        """Get grid position for Taxi (5x5 grid)."""
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        state = unwrapped.s
        return (state // 25, state % 25)
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """Taxi is a 5x5 grid."""
        return (5, 5)


class TaxiConfig(EnvironmentConfig):
    """Configuration for Taxi-v3 environment."""
    
    @property
    def observation_space_size(self) -> int:
        """Taxi has 500 states (5x5 grid, 5 passenger locations, 4 destinations)."""
        return 500
    
    @property
    def action_space_size(self) -> int:
        """Taxi has 6 discrete actions."""
        return 6
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """Taxi is a 5x5 grid."""
        return (5, 5)
    
    @property
    def state_manager_class(self):
        """Returns TaxiStateManager class."""
        return TaxiStateManager


class CartPoleStateManager(StateManager):
    """Example: State manager for CartPole (continuous, non-grid environment).
    
    NOT IMPLEMENTED - This is a template for future work.
    
    Note: CartPole has continuous state space, so grid operations don't apply.
    """
    
    @staticmethod
    def clone_state(env: gym.Env) -> Dict[str, Any]:
        """Clone state of CartPole environment."""
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        state_dict = {
            'state': copy.deepcopy(unwrapped.state),  # [x, x_dot, theta, theta_dot]
            'np_random_state': copy.deepcopy(unwrapped.np_random.bit_generator.state),
        }
        
        if hasattr(env, '_elapsed_steps'):
            state_dict['_elapsed_steps'] = env._elapsed_steps
        
        return state_dict
    
    @staticmethod
    def restore_state(env: gym.Env, state_dict: Dict[str, Any]) -> None:
        """Restore state of CartPole environment."""
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        unwrapped.state = copy.deepcopy(state_dict['state'])
        unwrapped.np_random.bit_generator.state = copy.deepcopy(state_dict['np_random_state'])
        
        if '_elapsed_steps' in state_dict and hasattr(env, '_elapsed_steps'):
            env._elapsed_steps = state_dict['_elapsed_steps']
    
    @staticmethod
    def get_state_info(env: gym.Env) -> Dict[str, Any]:
        """Get state info for CartPole (continuous state)."""
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        x, x_dot, theta, theta_dot = unwrapped.state
        
        return {
            'state': unwrapped.state,
            'x': x,
            'x_dot': x_dot,
            'theta': theta,
            'theta_dot': theta_dot,
        }
    
    @staticmethod
    def get_grid_position(env: gym.Env) -> Tuple[int, int]:
        """CartPole is not grid-based."""
        raise NotImplementedError("CartPole has continuous state space - no grid position")
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """CartPole is not grid-based."""
        return None


class CartPoleConfig(EnvironmentConfig):
    """Configuration for CartPole-v1 environment."""
    
    @property
    def observation_space_size(self) -> int:
        """CartPole has 4-dimensional continuous observation space."""
        return 4
    
    @property
    def action_space_size(self) -> int:
        """CartPole has 2 discrete actions (push left, push right)."""
        return 2
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        """CartPole is not grid-based."""
        return None
    
    @property
    def state_manager_class(self):
        """Returns CartPoleStateManager class."""
        return CartPoleStateManager


# ============================================================================
# HOW TO ADD A NEW ENVIRONMENT
# ============================================================================
#
# 1. Create a new StateManager subclass:
#
#    class MyEnvStateManager(StateManager):
#        @staticmethod
#        def clone_state(env):
#            # Return dict with all state information
#            pass
#        
#        @staticmethod
#        def restore_state(env, state_dict):
#            # Restore env to previous state
#            pass
#        
#        @staticmethod
#        def get_state_info(env):
#            # Return human-readable state info
#            pass
#        
#        @staticmethod
#        def get_grid_position(env):
#            # Return (row, col) if grid-based
#            pass
#        
#        @property
#        def grid_shape(self):
#            # Return (rows, cols) or None
#            pass
#
# 2. Create a Config subclass:
#
#    class MyEnvConfig(EnvironmentConfig):
#        @property
#        def observation_space_size(self): ...
#        
#        @property
#        def action_space_size(self): ...
#        
#        @property
#        def grid_shape(self): ...
#        
#        @property
#        def state_manager_class(self): ...
#
# 3. Register the environment:
#
#    from counterfactual_rl.environments import StateManagerFactory
#    StateManagerFactory.register("MyEnv-v0", MyEnvStateManager, MyEnvConfig)
#
# 4. Use in your code:
#
#    from counterfactual_rl.environments import StateManagerFactory
#    
#    env_id = "MyEnv-v0"
#    state_manager = StateManagerFactory.get_state_manager(env_id)
#    config = StateManagerFactory.get_config(env_id)
#
# ============================================================================
