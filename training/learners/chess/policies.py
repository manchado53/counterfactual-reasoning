"""
ChessQNetwork — Q-network for Gardner chess (5x5).

Takes a flattened (2875,) board observation and outputs (1, 1225) Q-values.
The output shape (1, 1225) matches the (n_agents, actions_per_agent) convention
used by the rest of the training code, with n_agents=1 for chess.

Architecture:
    Reshape (2875,) -> (5, 5, 115)
    Conv(32, 3x3, same) -> relu
    Conv(64, 3x3, same) -> relu
    Conv(64, 1x1)       -> relu
    Flatten             -> (1600,)
    Dense(hidden_dim)   -> relu
    Dense(256)          -> relu
    Dense(1225)         -> expand_dims(-2) -> (1, 1225)
"""

import jax.numpy as jnp
import flax.linen as nn


class ChessQNetwork(nn.Module):
    hidden_dim: int = 512
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, flat_obs):
        """
        Args:
            flat_obs: (..., 2875) float32

        Returns:
            Q-values: (..., 1, 1225)
        """
        # Reshape to spatial board representation
        x = flat_obs.reshape((*flat_obs.shape[:-1], 5, 5, 115))

        # Convolutional front-end
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)

        x = nn.Conv(features=64, kernel_size=(1, 1))(x)
        x = nn.relu(x)

        # Flatten spatial dims: (5, 5, 64) -> (1600,)
        x = x.reshape((*flat_obs.shape[:-1], -1))

        # MLP
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)

        x = nn.Dense(256)(x)
        x = nn.relu(x)

        q = nn.Dense(1225)(x)  # (..., 1225)

        # Add agent dimension: (..., 1225) -> (..., 1, 1225)
        return jnp.expand_dims(q, axis=-2)
