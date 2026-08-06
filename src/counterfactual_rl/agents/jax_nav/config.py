"""
Default config for JaxNav agents.

Mirrors ``agents/frozen_lake/config.py``: every training and CCE knob keeps the
same name and meaning. Only the environment block differs (JaxNav robot-nav
params instead of FrozenLake's map/slippery), plus one new CCE knob
(``cf_rollout_temperature``) — see the note there.
"""

DEFAULT_CONFIG = {
    # --- Environment (JaxNav) ---
    'scenario': None,             # e.g. 'SingleNav1' for a fixed reproducible map; None -> random maps
    'map_id': 'Grid-Rand-Poly',   # random-map generator (used when scenario is None)
    'map_size': (11, 11),
    'fill': 0.3,                  # obstacle density for random maps
    'goal_radius': 0.5,
    'goal_rew': 1.0,              # FrozenLake-style +1 on reaching the goal
    'coll_rew': 0.0,              # 0.0 -> collision is a 0-reward terminal (FrozenLake "hole" analog)
    'max_steps': 200,
    'sparse_reward': True,        # zero the dense shaping -> return>0 <=> reached goal

    'algorithm': 'consequence-dqn',
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 15000,
    'alpha': 0.0003,
    'batch_size': 128,
    'buffer_capacity': 100000,
    'target_update_freq': 2000,   # in env steps (converted to grad steps in the vectorized loop)
    'n_steps_per_update': 16,     # replay ratio = batch/this = 8 (FrozenLake's; 4 gave ~32 -> unstable)
    'double_dqn': True,           # decouple action-selection from evaluation -> curbs Q overestimation
    'hidden_dim': 128,
    'n_layers': 3,
    'n_episodes': 30000,
    'eval_interval': 300,
    'eval_episodes': 100,
    'save_every': 1000,
    'n_checkpoints': 100,
    'seed': 0,
    'PER_parameters': {'eps': 0.01, 'beta': 0.25, 'maximum_priority': 1.0},

    # --- Consequence-weighted PER (identical knobs to FrozenLake) ---
    'mu': 0.5,
    'priority_mixing': 'additive',
    'mu_c': 1.0,
    'mu_delta': 1.0,
    'score_interval': 100,
    'n_score_sample': 128,
    'consequence_metric': 'total_variation',
    'consequence_aggregation': 'weighted_mean',
    'cf_horizon': 20,
    'cf_n_rollouts': 20,
    'cf_gamma': 0.99,
    # NEW for JaxNav. FrozenLake's per-action return spread comes from stochastic slip;
    # single-agent JaxNav transitions are DETERMINISTIC, so a greedy rollout collapses each
    # action's return distribution to a point and TV degrades to a coarse 0/1. A stochastic
    # rollout policy restores graded spread. Softmax temperature on the greedy Q:
    #   0.0 -> pure argmax (greedy ablation);  higher -> softer, more exploratory rollouts.
    'cf_rollout_temperature': 0.5,

    # --- Vectorized training (lax.scan + vmap) ---
    'vectorized': True,           # JaxNav is deep RL -> the vectorized trainer is the workhorse
    'n_envs': 128,
    'collect_steps': 32,
}
