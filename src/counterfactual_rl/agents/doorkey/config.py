DEFAULT_CONFIG = {
    # --- Environment (DoorKey) ---
    'layout_name': '6x6',
    # slip_prob: 0.0 = deterministic (Claim 2 sample-efficiency).
    #            > 0  = stochastic actuator (Claim 1 needs this so CCE's total-variation
    #                   signal is non-degenerate — mirrors FrozenLake's slippery Claim 1).
    'slip_prob': 0.0,

    'algorithm': 'consequence-dqn',
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 7500,
    'alpha': 0.001,
    'batch_size': 32,
    'buffer_capacity': 100000,
    'target_update_freq': 200,
    'n_steps_per_update': 4,
    'hidden_dim': 64,
    'n_layers': 2,
    'n_episodes': 15000,
    # DoorKey's only terminal is the goal (no holes/timeout in the tabular MDP), so we
    # truncate episodes at this length in the training loop — otherwise a random early
    # policy never finishes an episode and epsilon never decays (deadlock). Keep it
    # BELOW collect_steps so truncation fires within a chunk (episode-length carry resets
    # per chunk); optimal path is ~11 steps, so 50 is ample headroom.
    'max_episode_steps': 50,
    'eval_interval': 200,
    'eval_episodes': 100,
    'save_every': 500,
    'n_checkpoints': 100,
    'seed': 0,
    'PER_parameters': {'eps': 0.01, 'beta': 0.25, 'maximum_priority': 1.0},

    # --- Consequence-weighted PER (CCE) ---
    'mu': 0.5,
    'priority_mixing': 'additive',
    'mu_c': 1.0,
    'mu_delta': 1.0,
    'score_interval': 100,
    'n_score_sample': 128,
    'consequence_metric': 'total_variation',
    'consequence_aggregation': 'weighted_mean',
    # cf_horizon must cover a full episode; DoorKey-6x6 optimal path is ~11 steps,
    # 60 comfortably covers slip detours.
    'cf_horizon': 60,
    'cf_n_rollouts': 20,
    'cf_gamma': 0.99,

    # --- Vectorized training (lax.scan + vmap) ---
    'vectorized': False,
    'n_envs': 256,
    'collect_steps': 128,
}
