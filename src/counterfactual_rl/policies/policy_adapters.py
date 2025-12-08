"""
Policy adapters for integrating custom RL implementations with CounterfactualAnalyzer.

This module provides adapters to make custom policy implementations compatible
with the standard interface expected by CounterfactualAnalyzer.
"""

import torch
import numpy as np


class CustomPPOAdapter:
    """
    Adapts custom PyTorch PPO implementation to Stable-Baselines3 interface.
    
    Provides .predict() method expected by CounterfactualAnalyzer while
    handling action masking for SMAC environments.
    
    Example:
        >>> ppo = PPO(state_size, n_agents * n_actions, ...)
        >>> adapter = CustomPPOAdapter(ppo, smac_env)
        >>> actions, _ = adapter.predict(obs, deterministic=True)
    """
    
    def __init__(self, ppo_model, env):
        """
        Initialize adapter.
        
        Args:
            ppo_model: Custom PPO model with get_action_prob() and get_value() methods
            env: SMAC environment (needed for action masking and agent info)
        """
        self.ppo = ppo_model
        self.env = env
        # Get environment info from SMAC
        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
    
    def predict(self, obs, deterministic=True):
        """
        Predict actions for all agents given observation.
        
        Args:
            obs: Global state observation (for centralized policy)
            deterministic: If True, use argmax (greedy). If False, sample from distribution.
        
        Returns:
            (actions, state) where:
                - actions: np.ndarray of shape (n_agents,) with integer actions
                - state: None (not used, included for SB3 compatibility)
        """
        with torch.no_grad():
            # Get logits from policy
            logits = self.ppo.get_action_prob(obs)  # Shape: (n_agents * n_actions,)
            logits = logits.view(self.n_agents, self.n_actions)  # Shape: (n_agents, n_actions)
            
            # Apply action masking
            avail_actions = torch.tensor(self.env.get_avail_actions())  # Shape: (n_agents, n_actions)
            masked_logits = logits + (avail_actions - 1) * 1e10  # Mask unavailable actions
            
            if deterministic:
                # Greedy action selection
                actions = masked_logits.argmax(dim=1)
            else:
                # Stochastic sampling from distribution
                dist = torch.distributions.Categorical(logits=masked_logits)
                actions = dist.sample()
            
            # Convert to numpy
            actions_np = actions.cpu().numpy()
        
        return actions_np, None


class RandomPolicy:
    """
    Random policy for baseline testing and debugging.
    
    Samples random valid actions from the action space, respecting action masking.
    Useful for testing the counterfactual analysis pipeline without a trained model.
    
    Example:
        >>> random_policy = RandomPolicy(smac_env)
        >>> actions, _ = random_policy.predict(obs, deterministic=False)
    """
    
    def __init__(self, env):
        """
        Initialize random policy.
        
        Args:
            env: SMAC environment (needed for action masking)
        """
        self.env = env
        # Get environment info from SMAC
        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
    
    def predict(self, obs, deterministic=False):
        """
        Sample random valid actions for all agents.
        
        Args:
            obs: Observation (not used, but included for interface compatibility)
            deterministic: Not used for random policy (always stochastic)
        
        Returns:
            (actions, state) where:
                - actions: np.ndarray of shape (n_agents,) with random valid actions
                - state: None (not used)
        """
        actions = []
        
        # Sample random valid action for each agent
        for agent_id in range(self.n_agents):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            
            # Ensure avail_actions is an array (handle scalar case)
            avail_actions = np.atleast_1d(avail_actions)
            
            # Find valid action indices
            valid_action_ids = np.where(avail_actions == 1)[0]
            
            if len(valid_action_ids) > 0:
                action = np.random.choice(valid_action_ids)
            else:
                print("No valid actions found for agent", agent_id)
                print(avail_actions)
                # Fallback: if no valid actions, choose action 0 (no-op)
                action = 0
            
            actions.append(action)
        
        return np.array(actions, dtype=np.int64), None
