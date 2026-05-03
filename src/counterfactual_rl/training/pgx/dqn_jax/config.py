"""
Default hyperparameter config for Gardner chess DQN training.

Key differences from SMAX defaults:
  gamma=0.99        sparse reward -> need long-horizon credit
  M=200000          longer games fill buffer slower
  C=1000            target update less frequent than SMAX
  cf_horizon=10     each rollout step = white+black pair (2 half-moves)
  cf_n_rollouts=16  opponent inference is expensive per rollout
"""

DEFAULT_CHESS_CONFIG = {
    'seed': 0,
    'env_name': 'gardner_chess',

    # DQN
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 20000,
    'alpha': 0.0001,
    'hidden_dim': 512,
    'use_layer_norm': True,

    # Replay buffer
    'M': 200000,
    'B': 64,

    # Update frequencies
    'C': 1000,
    'n_steps_for_Q_update': 4,

    # Prioritized experience replay
    'PER_parameters': {
        'eps': 0.01,
        'beta': 0.4,
        'maximum_priority': 1.0,
    },

    # Training
    'n_episodes': 100000,
    'save_every': 1000,

    # Evaluation
    'eval_interval': 500,
    'eval_episodes': 50,

    # Algorithm selection: 'dqn-uniform', 'dqn', or 'consequence-dqn'
    'algorithm': 'consequence-dqn',

    # Consequence-weighted PER (Algorithm 2)
    'mu': 0.5,
    'priority_mixing': 'additive',
    'mu_c': 1.0,
    'mu_delta': 1.0,
    'score_interval': 200,
    'n_score_sample': 128,
    'consequence_metric': 'wasserstein',
    'consequence_aggregation': 'weighted_mean',

    # Counterfactual rollouts
    'cf_horizon': 10,       # steps; each step = white move + opponent response
    'cf_n_rollouts': 16,
    'cf_top_k': 10,
    'cf_gamma': 0.99,

    # Vectorized episode collection (lax.scan + vmap)
    'n_envs': 256,         # parallel environments; pgx paper uses 1024 on A100, 256 safe on T4
    'collect_steps': 256,  # scan length per chunk; covers max game length (256 half-move pairs)

    # Diagnostics
    'diagnostics_enabled': False,
    'diagnostics_plot_interval': 100,

    # Game recording: save an SVG animation every N chunks (None = disabled)
    'record_interval': None,
}
