"""
Connect4QNetwork — Q-network for Connect Four (6×7 board).

Takes a flattened (84,) board observation and outputs (1, 7) Q-values.
The output shape (1, 7) matches the (n_agents, actions_per_agent) convention
used by the rest of the training code, with n_agents=1.

Architecture (matched to neoyung, 96% vs random):
    Reshape (84,) -> (6, 7, 2)
    Conv(32, 5×5, SAME) -> leaky_relu
    Conv(64, 5×5, SAME) -> leaky_relu
    Conv(32, 1×1)       -> leaky_relu
    Flatten             -> (32*6*7 = 1344,)
    Dense(256)          -> leaky_relu
    Dense(128)          -> leaky_relu
    Dense(7)            -> expand_dims(-2) -> (1, 7)

Key design choices vs original:
  LeakyReLU: ReLU on sparse boolean inputs (3-14% non-zero) causes zero conv-weight
    gradients — pre-activations go negative, ReLU kills them, gradient is 0.
    LeakyReLU keeps a small gradient (slope=0.01) for negative pre-activations,
    allowing weights to update even when most board cells are empty.
  5×5 kernels: each piece influences a 5×5 = 25-cell output neighborhood (vs 9
    with 3×3), giving 2.8× more weight coverage per piece on sparse boards.
  3 conv layers: with 5×5 same-padding, receptive field is 5→9→13 cells —
    covers the full 6×7 board by layer 3, matching neoyung's effective coverage.
"""

import jax.numpy as jnp
import flax.linen as nn


class Connect4QNetwork(nn.Module):
    hidden_dim: int = 256
    use_layer_norm: bool = False  # unused; kept for interface compatibility

    @nn.compact
    def __call__(self, flat_obs):
        """
        Args:
            flat_obs: (..., 84) float32

        Returns:
            Q-values: (..., 1, 7)
        """
        x = flat_obs.astype(jnp.float32)

        # Reshape to spatial board representation: (6, 7, 2)
        x = x.reshape((*flat_obs.shape[:-1], 6, 7, 2))

        # Convolutional front-end — 5×5 kernels, LeakyReLU
        x = nn.Conv(features=32, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.leaky_relu(x)

        x = nn.Conv(features=64, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.leaky_relu(x)

        x = nn.Conv(features=32, kernel_size=(1, 1))(x)
        x = nn.leaky_relu(x)

        # Flatten spatial dims: (6, 7, 32) -> (1344,)
        x = x.reshape((*flat_obs.shape[:-1], -1))

        # MLP head
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.leaky_relu(x)

        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)

        q = nn.Dense(7)(x)  # (..., 7)

        # Add agent dimension: (..., 7) -> (..., 1, 7)
        return jnp.expand_dims(q, axis=-2)
