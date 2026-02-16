"""Default configuration and hyperparameters for SMAX DQN."""

from datetime import datetime

DEFAULT_CONFIG = {
    # Runtime
    'backend': 'jax',  # 'jax' or 'pytorch'
    'seed': 0,

    # Environment
    'scenario': '3m',
    'obs_type': 'world_state',  # 'world_state' or 'concatenated'

    # DQN hyperparameters
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 20000,  # Linear decay over this many episodes
    'alpha': 0.0005,
    'hidden_dim': 256,

    # Replay buffer
    'M': 100000,  # Buffer size
    'B': 32,      # Batch size

    # Update frequencies
    'C': 500,                   # Target network update frequency
    'n_steps_for_Q_update': 4,  # Steps between Q updates

    # Prioritized Experience Replay
    'PER_parameters': {
        'eps': 0.01,
        'beta': 0.25,
        'maximum_priority': 1.0
    },

    # Training
    'n_episodes': 2000,
    'save_every': 500,

    # Periodic evaluation during training
    'eval_interval': None,    # Evaluate every N episodes (None = disabled)
    'eval_episodes': 20,      # Episodes per evaluation
}
