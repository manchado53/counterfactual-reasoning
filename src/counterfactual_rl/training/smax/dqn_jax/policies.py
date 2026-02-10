"""
Flax policy network for SMAX DQN.

Multi-headed MLP that takes global state and outputs Q-values for all agents.
Equivalent architecture to the PyTorch CentralizedQNetwork.
"""

import jax.numpy as jnp
import flax.linen as nn


class CentralizedQNetwork(nn.Module):
    """
    Centralized Q-Network for multi-agent control (Flax).

    Takes concatenated observations (global state) and outputs Q-values
    for each agent's action space via separate heads.

    Architecture:
        Global State -> Shared MLP Body (3 layers) -> Per-Agent Q-Value Heads
    """
    num_agents: int
    actions_per_agent: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_state):
        """
        Forward pass.

        Args:
            global_state: (..., obs_dim) array

        Returns:
            Q-values: (..., num_agents, actions_per_agent) array
        """
        # Shared body — 3 layer MLP
        x = nn.relu(nn.Dense(self.hidden_dim)(global_state))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))

        # Per-agent Q-value heads
        q_values = [nn.Dense(self.actions_per_agent)(x) for _ in range(self.num_agents)]
        return jnp.stack(q_values, axis=-2)  # (..., num_agents, actions_per_agent)
