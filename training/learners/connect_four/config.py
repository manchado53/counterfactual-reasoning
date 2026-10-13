ARMS = ("uniform", "per", "cce_mul", "cce_add", "cce_only")

RECIPE = dict(gamma=.99, alpha=.0003, hidden_dim=128, n_layers=3, batch_size=128,
              buffer_capacity=100000, n_steps_per_update=16, target_update_freq=2000,
              n_envs=256, collect_steps=32, epsilon_decay_episodes=62500,
              epsilon_start=1., epsilon_end=.05, cf_horizon=60, cf_n_rollouts=1,
              cf_gamma=.99, cf_rollout_temperature=0., cf_action_prob_temperature=.5,
              cf_score_chunk=1280, double_dqn=True, cf_bootstrap=True,
              explore_hold_steps=1, score_on_insert=True, refresh=False,
              consequence_metric="wasserstein", consequence_aggregation="weighted_mean",
              normalise_priorities="rank", gradient_clip=10., fixed_eval_seed=20260923,
              eval_episodes=1000, blocks=2000, eval_every_blocks=50,
              opponent="win-block-center-v1", opponent_ties=[3, 2, 4, 1, 5, 0, 6],
              PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1.))

def effective_config(arm, seed):
    cce = arm.startswith('cce')
    return dict(RECIPE, arm=arm, seed=seed, cce_enabled=cce,
                normalise_priorities='rank' if cce else 'none', score_on_insert=cce,
                mu=1. if arm == 'cce_only' else .5, mu_c=1., mu_delta=1.,
                priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive',
                active_features=dict(cce=cce, raw_td_per=arm == 'per',
                                     uniform_replay=arm == 'uniform', refresh=False),
                inactive_keys=[] if cce else ['cf_horizon', 'cf_n_rollouts', 'cf_gamma',
                    'cf_rollout_temperature', 'cf_action_prob_temperature', 'cf_score_chunk',
                    'cf_bootstrap', 'consequence_metric', 'consequence_aggregation',
                    'mu', 'mu_c', 'mu_delta', 'priority_mixing'])

config_for = effective_config
