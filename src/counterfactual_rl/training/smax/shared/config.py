"""Default configuration and hyperparameters for SMAX DQN."""

from datetime import datetime

DEFAULT_CONFIG = {
    # Runtime
    'backend': 'jax',  # 'jax' or 'pytorch'
    'seed': 0,

    # Environment
    'scenario': '3s5z',
    'obs_type': 'concatenated',  # 'world_state' or 'concatenated'

    # DQN hyperparameters
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 10000,  # Linear decay over this many episodes
    'alpha': 0.0005,
    'hidden_dim': 256,
    'n_body_layers': 3,
    'n_head_layers': 1,
    'use_layer_norm': False,

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
    'n_episodes': 30000,
    'save_every': 500,

    # Periodic evaluation during training
    'eval_interval': 100,    # Evaluate every N episodes (None = disabled)
    'eval_episodes': 100,      # Episodes per evaluation

    # Algorithm selection
    'algorithm': 'consequence-dqn',              # 'dqn-uniform', 'dqn', or 'consequence-dqn'

    # Consequence-weighted PER (Algorithm 2) — only used when algorithm='consequence-dqn'
    'mu': 0.5,                        # Weight: 0=pure TD, 1=pure consequence
    'score_interval': 100,              # Score every N Q-updates (1 = matches paper)
    'n_score_sample': 256,            # B^C_est: transitions scored per pass
    'consequence_metric': 'wasserstein',  # 'kl_divergence''jensen_shannon''total_variation''wasserstein'
    'consequence_aggregation': 'weighted_mean',
    'cf_horizon': 30,                 # Rollout horizon
    'cf_n_rollouts': 30,              # Rollouts per action
    'cf_top_k': 10,                   # Top-K actions from beam search
    'cf_gamma': 0.95,                 # Discount factor for rollouts

    # Diagnostics
    'diagnostics_plot_interval': 50,  # Generate diagnostic plot every N scoring passes
}

# Per-scenario architecture/training presets.
# Applied as: DEFAULT_CONFIG < SCENARIO_PRESET < CONFIG_OVERRIDES_B64
SCENARIO_PRESETS = {
    # Small (3-5 allies)
    '3m':        {'hidden_dim': 128, 'n_body_layers': 2, 'n_head_layers': 1, 'use_layer_norm': False, 'B': 32, 'alpha': 0.0005},
    '2s3z':      {'hidden_dim': 128, 'n_body_layers': 3, 'n_head_layers': 1, 'use_layer_norm': False, 'B': 32, 'alpha': 0.0005},
    '3s_vs_5z':  {'hidden_dim': 128, 'n_body_layers': 3, 'n_head_layers': 1, 'use_layer_norm': False, 'B': 32, 'alpha': 0.0005},
    '5m_vs_6m':  {'hidden_dim': 192, 'n_body_layers': 3, 'n_head_layers': 1, 'use_layer_norm': False, 'B': 32, 'alpha': 0.0005},
    # Medium (6-8 allies, mixed units)
    '3s5z':      {'hidden_dim': 256, 'n_body_layers': 3, 'n_head_layers': 2, 'use_layer_norm': True, 'B': 64, 'alpha': 0.0003},
    '8m':        {'hidden_dim': 256, 'n_body_layers': 3, 'n_head_layers': 1, 'use_layer_norm': True, 'B': 64, 'alpha': 0.0003},
    '3s5z_vs_3s6z': {'hidden_dim': 256, 'n_body_layers': 3, 'n_head_layers': 2, 'use_layer_norm': True, 'B': 64, 'alpha': 0.0003},
    # Large (10+ allies)
    '10m_vs_11m': {'hidden_dim': 512, 'n_body_layers': 4, 'n_head_layers': 2, 'use_layer_norm': True, 'B': 128, 'alpha': 0.0001},
    '25m':        {'hidden_dim': 512, 'n_body_layers': 4, 'n_head_layers': 2, 'use_layer_norm': True, 'B': 128, 'alpha': 0.0001},
}
