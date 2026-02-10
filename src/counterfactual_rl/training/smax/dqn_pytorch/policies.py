"""
Policy networks for SMAX DQN.

Multi-headed MLP that takes global state and outputs Q-values for all agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralizedQNetwork(nn.Module):
    """
    Centralized Q-Network for multi-agent control.

    Takes concatenated observations (global state) and outputs Q-values
    for each agent's action space via separate heads.

    Architecture:
        Global State -> Shared MLP Body -> Per-Agent Q-Value Heads
    """

    def __init__(self, obs_dim: int, num_agents: int, actions_per_agent: int, hidden_dim: int = 256):
        """
        Args:
            obs_dim: Dimension of the global state (concatenated observations)
            num_agents: Number of agents to control
            actions_per_agent: Number of actions available to each agent
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.actions_per_agent = actions_per_agent
        self.hidden_dim = hidden_dim

        # Shared body - 3 layer MLP
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Per-agent Q-value heads
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, actions_per_agent)
            for _ in range(num_agents)
        ])

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            global_state: Tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns:
            Q-values of shape (batch, num_agents, actions_per_agent)
            or (num_agents, actions_per_agent) if no batch dimension
        """
        # Handle single state (no batch dimension)
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Shared body
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Per-agent Q-values
        q_values = [head(x) for head in self.q_heads]
        q_values = torch.stack(q_values, dim=1)  # (batch, num_agents, actions)

        if squeeze_output:
            q_values = q_values.squeeze(0)  # (num_agents, actions)

        return q_values
