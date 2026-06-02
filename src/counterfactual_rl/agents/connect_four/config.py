"""
Default hyperparameter config for Connect Four DQN training.

Matched to neoyung (96% win rate vs random) — key differences from broken v1:
  use_layer_norm=False  LayerNorm on sparse boolean obs caused Q-value divergence
  alpha=0.001           10× larger lr (was 0.0001); matches neoyung and FL
  gamma=0.999           longer credit horizon for sparse end-of-game rewards
  B=256                 larger batch for more stable gradients (was 64)
  C=1_000              target_freq_q = 1000 // 4 = 250 grad steps ≈ neoyung's 210
  n_steps_for_Q_update=4   UTD=1/4; 10752 // 4 = 2688 gradient steps per chunk
  score_interval=2500   ~1 scoring pass/chunk (2688 ÷ 2500 ≈ 1.07×)
  exploration_fraction=0.5  chunk-based epsilon decay (no episode counting)
  eval_interval=1       eval every chunk → 300 checkpoints, maximum curve resolution
  eval_episodes=200     ±6pp CI at p=0.75; games fast enough that cost is negligible
"""

DEFAULT_CONNECT4_CONFIG = {
    'seed': 0,
    'env_name': 'connect_four',

    # DQN
    'gamma': 0.999,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'exploration_fraction': 0.5,  # chunk-based decay; no epsilon_decay_episodes
    'alpha': 0.001,
    'hidden_dim': 256,
    'use_layer_norm': False,

    # Replay buffer
    'M': 100_000,
    'B': 256,

    # Update frequencies
    'C': 1_000,                # target_freq_q = 1000 // 4 = 250 grad steps between syncs
    'n_steps_for_Q_update': 4, # UTD=1/4; 2688 grad steps/chunk (was 64 → 168 steps/chunk)

    # Prioritized experience replay
    'PER_parameters': {
        'eps': 0.01,
        'beta': 0.25,
        'maximum_priority': 1.0,
    },

    # Training
    'n_chunks': 700,
    'save_every': 10,
    'n_checkpoints': 10,

    # Opponent for both training collection and CF rollouts: 'random', 'rule_based', or 'mcts'
    'opponent': 'random',
    'mcts_n_sims': 32,   # MCTS simulations per opponent move (used when opponent='mcts')

    # Evaluation (random opponent only — no pgx baseline for connect_four)
    'eval_interval': 1,
    'eval_episodes': 200,
    'eval_opponent': 'random',

    # Algorithm selection: 'dqn-uniform', 'dqn', or 'consequence-dqn'
    'algorithm': 'consequence-dqn',

    # Consequence-weighted PER (Algorithm 2)
    'mu': 0.25,
    'priority_mixing': 'additive',
    'mu_c': 1.0,
    'mu_delta': 1.0,
    'score_interval': 1250,    # ~2 scoring passes/chunk (2688 Q-updates/chunk ÷ 1250)
    'n_score_sample': 128,     # 128×2=256 transitions/chunk — same coverage, half the memory per pass
    'consequence_metric': 'total_variation',
    'consequence_aggregation': 'weighted_mean',

    # Counterfactual rollouts
    'cf_horizon': 42,      # 42 half-moves = max Connect Four game length → always reaches terminal
    'cf_n_rollouts': 30,
    'cf_top_k': 7,         # all 7 columns — complete counterfactual coverage
    'cf_gamma': 0.99,

    # Vectorized episode collection (lax.scan + vmap)
    # Each scan step = agent move + opponent random move = 1 stored transition
    'n_envs': 256,
    'collect_steps': 42,   # 42 full-round pairs = ~2 complete games per env/chunk

    # Diagnostics
    'diagnostics_enabled': False,
    'diagnostics_plot_interval': 100,
}
