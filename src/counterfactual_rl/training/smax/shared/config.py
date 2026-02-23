"""Default configuration and hyperparameters for SMAX DQN."""

from datetime import datetime

DEFAULT_CONFIG = {
    # Runtime
    'backend': 'jax',  # 'jax' or 'pytorch'
    'seed': 0,

    # Environment
    'scenario': '3m',
    'obs_type': 'concatenated',  # 'world_state' or 'concatenated'

    # DQN hyperparameters
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 100,  # Linear decay over this many episodes
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
    'n_episodes': 1000,
    'save_every': 500,

    # Periodic evaluation during training
    'eval_interval': 100,    # Evaluate every N episodes (None = disabled)
    'eval_episodes': 100,      # Episodes per evaluation

    # Algorithm selection
    'algorithm': 'consequence-dqn',              # 'dqn' or 'consequence-dqn'

    # Consequence-weighted PER (Algorithm 2) — only used when algorithm='consequence-dqn'
    'mu': 0.5,                        # Weight: 0=pure TD, 1=pure consequence
    'score_interval': 20,              # Score every N Q-updates (1 = matches paper)
    'n_score_sample': 256,            # B^C_est: transitions scored per pass
    'consequence_metric': 'wasserstein',  # 'kl_divergence''jensen_shannon''total_variation''wasserstein'
    'consequence_aggregation': 'weighted_mean',
    'cf_horizon': 30,                 # Rollout horizon
    'cf_n_rollouts': 48,              # Rollouts per action
    'cf_top_k': 20,                   # Top-K actions from beam search
    'cf_gamma': 0.95,                 # Discount factor for rollouts
}
