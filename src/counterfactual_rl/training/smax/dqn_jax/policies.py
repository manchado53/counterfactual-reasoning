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
        Global State -> Shared MLP Body (n_body_layers) -> Per-Agent Q-Value Heads
    """
    num_agents: int
    actions_per_agent: int
    hidden_dim: int = 256
    n_body_layers: int = 3
    n_head_layers: int = 1
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, global_state):
        """
        Forward pass.

        Args:
            global_state: (..., obs_dim) array

        Returns:
            Q-values: (..., num_agents, actions_per_agent) array
        """
        # Shared body
        x = global_state
        for _ in range(self.n_body_layers):
            x = nn.Dense(self.hidden_dim)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        # Per-agent Q-value heads
        q_values = []
        for _ in range(self.num_agents):
            h = x
            for _ in range(self.n_head_layers - 1):
                h = nn.Dense(self.hidden_dim // 2)(h)
                if self.use_layer_norm:
                    h = nn.LayerNorm()(h)
                h = nn.relu(h)
            h = nn.Dense(self.actions_per_agent)(h)
            q_values.append(h)

        return jnp.stack(q_values, axis=-2)  # (..., num_agents, actions_per_agent)
