"""
Default config for the neural-CVRP (RP-DQN) agent.

Hyperparameters marked (RP-DQN) are theirs, recovered from ``main.py`` in
github.com/inboxedshoe/RP-DQN -- the paper publishes none of them. Where the
paper's prose and the shipped code disagree, we follow the CODE, because the code
is what produced the 6.36 mean tour length in their Table 1:

  * exploration -- the paper says epsilon-greedy; ``main.py`` ships
    ``strategy: 'boltzman'``, start 1.0, end 0.02, linear over 50% of training.
  * target network -- ``target_network_update: 0.999`` is a Polyak/soft update
    every gradient step, not the hard every-C-steps sync the rest of this repo uses.
  * ``normalization: 'no'`` -- no BatchNorm and no LayerNorm in the encoder. This
    also happens to match the repo's own sparse-boolean-observation gotcha, where
    LayerNorm zeroed the gradient and Q diverged.

UNITS WARNING. ``n_episodes`` counts INDIVIDUAL episodes, same as every other
agent here. Their 250,000 is also individual episodes, collected 64 at a time
(``batchsize_collect: 64``) -- it is NOT 250,000 batches. Cross-checked against
their Fig. 2, whose x-axis tops out near 20M samples: 250,000 episodes x ~60
decisions on MDVRP50 lands there, while 250,000 *batches* would be ~60x too many.
At n=20 an episode is ~25 decisions, so a full run is ~6.3M env steps and
~13k gradient steps.
"""

DEFAULT_CONFIG = {
    # --- Environment ---
    'n_customers': 20,          # (RP-DQN) 20 / 50 / 100; we run 20 -- 4.26% headroom
    'capacity': None,           # None -> DataGenerator's rule: 30 at n<=20
    'max_steps': None,          # None -> 2 * n_customers + 2

    # Mean tour length of the OPTIMAL solver on their n=20 test set (Table 1,
    # Gurobi = 6.1). Only used to turn tour length into an increasing 'win_rate'
    # so the shared claim-2 analysis works; it never touches training.
    'optimal_ref': 6.10,

    'algorithm': 'consequence-dqn',   # dqn-uniform | dqn (=PER) | consequence-dqn

    # --- Q-learning (RP-DQN) ---
    'gamma': 1.0,               # discount_factor: 1 -> return == negative tour length
    'alpha': 1e-4,              # lr
    'batch_size': 1024,         # batchsize
    'buffer_capacity': 2 ** 14, # buffer_capacity
    'buffer_min_samples': 2 ** 13,
    'double_dqn': True,         # "Double DQN with N-step returns", N=1
    'n_step': 1,                # n: 1

    # Polyak target update, applied every gradient step (RP-DQN).
    # target_update_freq is kept only so the inherited hard-sync path still works
    # if target_tau is set to None.
    'target_tau': 0.999,
    'target_update_freq': 2000,

    # --- Exploration: Boltzmann, linear over the first half (RP-DQN) ---
    'exploration': 'boltzmann',   # 'boltzmann' | 'epsilon'
    'temp_start': 1.0,
    'temp_end': 0.02,
    'epsilon_start': 1.0,         # used only when exploration == 'epsilon'
    'epsilon_end': 0.02,
    'exploration_fraction': 0.5,  # linear_percentage

    # --- Network: Kool-style attention encoder, no normalization (RP-DQN) ---
    'embed_dim': 128,           # embedding_dimension
    'n_layers': 3,              # L = 3 for VRPs
    'n_heads': 8,
    'ff_dim': 512,              # intermediate_dimension
    'use_layer_norm': False,    # normalization: 'no'

    # --- Training loop ---
    # n_envs mirrors batchsize_collect=64. collect_steps=32 is a bit over one
    # n=20 episode, so a chunk retires ~80 episodes and fires ~4 gradient steps --
    # their learn_iter_per_episode=4 at a matching replay ratio (~2.5).
    'n_episodes': 250000,
    'n_envs': 64,
    'collect_steps': 32,
    'n_steps_per_update': 512,
    'vectorized': True,

    'eval_interval': 2500,
    'eval_episodes': 512,
    'save_every': 10000,
    'n_checkpoints': 50,
    'seed': 0,

    'PER_parameters': {'eps': 0.01, 'beta': 0.4, 'maximum_priority': 1.0},  # beta_start: 0.4

    # --- Early stopping: OFF. ---
    # The whole point of this benchmark is the last few percent of tour length, and
    # a plateau stopper would cut exactly the tail we are trying to measure.
    'early_stop_win_rate': None,
    'early_stop_patience': None,
    'early_stop_min_delta': 0.0,
    'early_stop_smooth_window': 5,
    'early_stop_min_episodes': None,

    # --- CCE ---
    # wasserstein, NOT total_variation. Their CVRP is deterministic with a continuous
    # distance reward, which is exactly where TV degenerates: metrics.py returns a
    # hardcoded 1.0 for any two distinct constants, so every state ties at the ceiling
    # and rho is undefined. Measured on the merged Claim-1 checkpoints, Wasserstein
    # beat TV at every training stage (rho .665/.869/.844 vs .576/.765/.741).
    'consequence_metric': 'wasserstein',
    'consequence_aggregation': 'weighted_mean',
    'priority_mixing': 'multiplicative',
    'mu': 0.5, 'mu_c': 1.0, 'mu_delta': 1.0,

    # cf_n_rollouts=1 is EXACT here, not an approximation: the env is deterministic
    # and the continuation policy is greedy (temperature 0), so N rollouts are
    # bit-identical. Verified max|diff| 0.00e+00 on the routing checkpoints.
    'cf_n_rollouts': 1,
    'cf_rollout_temperature': 0.0,
    'cf_horizon': 10,
    'cf_gamma': 1.0,            # match their gamma; 0.99 would score "same stops,
                                # different order" differently
    'cf_bootstrap': True,       # horizon 10 covers only ~40% of a ~25-decision
                                # episode, so truncation bias matters here (#7)

    # Coverage. E[times scored] = n_score_sample / (score_interval * n_steps_per_update).
    # FrozenLake ran 128/(100*4) = 32% and won C2; JaxNav ran 64/(250*16) = 1.6% and
    # was a null because almost nothing in the buffer ever held a measured score.
    # Here 512/(4*512) = 25% LIFETIME. That is not what the sampler sees: measured
    # on JaxNav (job 274677, per-entry scored_at), the instantaneous scored fraction
    # in a full buffer is about half the lifetime figure, following
    # 1 - (1 - e^-L)/L. At L=0.25 that is ~11.5% instantaneous, ramping with entry
    # age. Quote the instantaneous number, not the 25%.
    # Still ~13x JaxNav's 0.90%, and the ess_frac 0.70-0.74 seen in smoke runs says
    # CCE is genuinely prioritising here. Raising this to hit 20% instantaneous would
    # need n_score_sample ~1024 (L~0.46) and doubles scoring cost -- decide from Gate 0.
    #
    # Do NOT raise this blindly: scoring cost is n_score_sample x n_actions x
    # cf_horizon TRANSFORMER forward passes per pass -- 512*21*10 = 107k against the
    # 2048 the chunk collected, so CCE is expected to dominate wall-clock here in a
    # way it never did on a tabular env. Gate 0 measures the real number and the
    # sweep is sized from it.
    'score_interval': 4,
    'n_score_sample': 512,
    'diagnostics_enabled': False,
}
