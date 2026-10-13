import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ARMS = ('uniform', 'per', 'cce_mul', 'cce_add', 'cce_only')

from .source_recipe import RECIPE

def config_for(arm, seed):
    row = next(r for r in RECIPE['rows'] if r['arm'] == arm and r['seed'] == seed)
    expected = dict(gamma=.99, alpha=.0003, hidden_dim=128, n_layers=3,
        batch_size=128, buffer_capacity=100000, target_update_freq=2000,
        n_steps_per_update=16, double_dqn=True, n_envs=256, collect_steps=32,
        epsilon_start=1., epsilon_end=.05, epsilon_decay_episodes=62500,
        fixed_eval_seed=20260923,
        PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1.))
    for name,value in expected.items():
        assert row['config'][name] == value, ('source recipe drift', name)
    cfg = dict(expected, seed=seed, arm=arm, cce_enabled=arm.startswith('cce'),
               optimizer='adam', gradient_clip=10., loss='importance-weighted-squared-td',
               target_copy_phase='after attempts 0,125,250,...',
               importance_weights='1/(N*p), no batch rescale')
    if cfg['cce_enabled']:
        cce = dict(consequence_metric='wasserstein', consequence_aggregation='weighted_mean',
            cf_horizon=60, cf_n_rollouts=1, cf_gamma=.99, cf_rollout_temperature=0,
            cf_bootstrap=True, score_on_insert=True, cf_score_chunk=1280,
            normalise_priorities='rank', cf_action_prob_temperature=.5,
            mu=1. if arm == 'cce_only' else .5, mu_c=1., mu_delta=1.,
            priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive')
        for name,value in cce.items():
            assert row['config'][name] == value, ('source CCE drift', name)
        cfg.update(cce, refresh_enabled=False, drift_probe_enabled=False)
    cfg.update(environment='custom-full-state-doorkey', layout_name='6x6', slip_prob=0.,
               random_start='fixed', max_episode_steps=50, eval_max_steps=50,
               explore_hold_steps=1, eval_episodes=1000, observation='one-hot-state-id',
               timeout_bootstrap=True, blocks=2000, eval_every_blocks=50)
    # All historical/inert keys remain in source_recipe.json, not runtime config.
    return cfg
