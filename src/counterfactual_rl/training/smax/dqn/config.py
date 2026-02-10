"""Default configuration and hyperparameters for SMAX DQN."""

DEFAULT_CONFIG = {
    # Environment
    'scenario': '3m',

    # DQN hyperparameters
    'gamma': 0.99,
    'epsilon': 0.1,
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
    'save_path': 'models/smax_dqn.pt',
}
