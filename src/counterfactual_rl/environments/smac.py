"""
SMAC Integration for Counterfactual RL.

This module provides the necessary components to use the StarCraft Multi-Agent Challenge (SMAC)
with the counterfactual reasoning framework.

It includes:
1. SmacStateManager: Handles state saving/restoring via Replay Strategy.
2. SmacConfig: Defines environment configuration.
3. CentralizedSmacWrapper: Wraps SMAC to look like a single-agent environment.
"""

import copy
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, List, Optional

from counterfactual_rl.environments.base import StateManager, EnvironmentConfig


class CentralizedSmacWrapper:
    """
    Wraps a multi-agent SMAC environment to appear as a single-agent environment.
    
    - Returns global state (full battlefield view) by default.
    - Accepts MultiDiscrete actions (array of actions, one per agent).
    - Tracks action history for the Replay Strategy.
    
    Note: Does not inherit from gym.Wrapper because SMAC uses old gym API.
    """
    def __init__(self, env, use_state=True):
        self.env = env
        self.action_history = []
        self._seed = None
        self.use_state = use_state  # If True, use global state; if False, use observations
        
        # Cache environment info
        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions_per_agent = env_info["n_actions"]
        
        # Create Gym-compatible action and observation spaces
        import gymnasium as gym
        # Use MultiDiscrete: each agent independently chooses from n_actions
        # Example for 8m: MultiDiscrete([14, 14, 14, 14, 14, 14, 14, 14])
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_actions_per_agent] * self.n_agents
        )
        
        # Get observation size by doing a dummy reset
        obs, _ = self.reset()
        obs_size = len(obs) if isinstance(obs, np.ndarray) else obs
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset environment and clear history."""
        self.action_history = []
        self._seed = seed
        
        # SMAC reset doesn't accept seed directly in all versions, 
        # but we store it for the StateManager to use if needed.
        obs, state = self.env.reset()
        
        # Return flattened observation and info
        return self._get_obs(), {}
        
    def step(self, actions):
        """
        Take actions for all agents.
        
        Args:
            actions: Array-like of shape (n_agents,) with integer actions.
                    Can be np.ndarray, torch.Tensor, or list.
        
        Returns:
            Standard Gym tuple (obs, reward, terminated, truncated, info)
        """
        # Convert to list for SMAC compatibility
        if isinstance(actions, np.ndarray):
            agent_actions = actions.tolist()
        elif hasattr(actions, 'cpu'):  # torch.Tensor
            import torch
            if torch.is_tensor(actions):
                agent_actions = actions.cpu().numpy().tolist()
            else:
                agent_actions = list(actions)
        else:
            agent_actions = list(actions)
        
        # Validate and fix invalid actions (action masking)
        agent_actions = self._mask_invalid_actions(agent_actions)
        
        # Record for history
        self.action_history.append(agent_actions)
        
        # Step environment
        reward, terminated, info = self.env.step(agent_actions)
        
        # Return standard Gym tuple
        truncated = False  # SMAC handles time limits via terminated usually
        return self._get_obs(), reward, terminated, truncated, info
    
    def _mask_invalid_actions(self, agent_actions: List[int]) -> List[int]:
        """Replace invalid actions with first available action."""
        valid_actions = []
        for agent_id, action in enumerate(agent_actions):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            if avail_actions[action] == 0:
                # Action is invalid, use first available action
                first_valid = np.argmax(avail_actions)  # Find first 1 in the array
                valid_actions.append(int(first_valid))
            else:
                valid_actions.append(action)
        return valid_actions
        
    def _get_obs(self):
        """Return global state or flattened observations."""
        if self.use_state:
            # Return global state (full battlefield view - like a human player)
            return self.env.get_state()
        else:
            # Return flattened observations (local view per agent)
            obs_list = self.env.get_obs()
            return np.concatenate(obs_list)


class SmacStateManager(StateManager):
    """
    State Manager for SMAC using the Replay Strategy.
    
    Since SC2 state cannot be directly saved/loaded, we:
    1. Save the history of actions.
    2. Restore by resetting and replaying the history.
    """
    
    @staticmethod
    def clone_state(env) -> Dict[str, Any]:
        """Save the action history."""
        # Ensure we are wrapping the right environment
        if not hasattr(env, 'action_history'):
            # Try unwrapping if it's nested
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'action_history'):
                history = env.unwrapped.action_history
            elif hasattr(env, 'env') and hasattr(env.env, 'action_history'):
                 history = env.env.action_history
            else:
                raise AttributeError("Environment must be wrapped with CentralizedSmacWrapper")
        else:
            history = env.action_history
            
        return {
            'action_history': copy.deepcopy(history),
            'seed': getattr(env, '_seed', None)
        }
    
    @staticmethod
    def restore_state(env, state_dict: Dict[str, Any]) -> None:
        """Restore state by REPLAYING history."""
        # 1. Reset
        env.reset(seed=state_dict.get('seed'))
        
        # 2. Replay
        history = state_dict['action_history']
        
        # Access the inner SMAC env directly for replay
        inner_env = env.env  # Assuming CentralizedSmacWrapper
        
        for agent_actions in history:
            inner_env.step(agent_actions)
            
        # Restore the wrapper's history to match the restored state
        env.action_history = copy.deepcopy(history)
        
        # IMPORTANT: After replay, we need to ensure the wrapper's observation is updated
        # This is critical for action masking to work correctly
        # The wrapper will call _get_obs() on the next step, which will get fresh data
    
    @staticmethod
    def get_state_info(env) -> Dict[str, Any]:
        """Get relevant state info."""
        # Access inner env to get unit info
        inner_env = env.env
        
        # Example: Get health of all units
        # Note: This depends on SMAC version/API availability
        info = {
            'state': 'running', # Placeholder
            'n_agents': inner_env.n_agents,
            'battles_won': inner_env.battles_won
        }
        return info
    
    @staticmethod
    def get_grid_position(env) -> Tuple[int, int]:
        """SMAC is not grid-based in the same way as FrozenLake."""
        return (0, 0) # Dummy return
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        return None


class SmacConfig(EnvironmentConfig):
    """Configuration for SMAC environments."""
    
    def __init__(self, map_name="3m"):
        self.map_name = map_name
        # Hardcoded for 3m for now, ideally should query env
        if map_name == "3m":
            self._n_agents = 3
            self._n_actions = 9
            self._obs_size = 30 # Approximate, depends on version
        else:
            # Default fallback
            self._n_agents = 3
            self._n_actions = 9
    
    @property
    def observation_space_size(self) -> int:
        # This should match the flattened observation size
        # For 3m: ~48 features per agent * 3 agents = 144 (example)
        # We'll need to check the actual env to be precise
        return 100 # Placeholder
    
    @property
    def action_space_size(self) -> int:
        # Joint action space: actions^agents
        return self._n_actions ** self._n_agents
    
    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        return None
    
    @property
    def state_manager_class(self):
        return SmacStateManager
