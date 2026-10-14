"""Frozen three-arm protocol; only replay differs within a seed."""
from .bank_v2 import load, digest, TRAIN_BANK

ARMS = ('uniform', 'per', 'cce_decay')


def config_for(arm, seed):
    if arm not in ARMS or type(seed) is not int or seed not in range(5):
        raise ValueError('Unknown arm or seed')
    _, bank_hash = load()
    return dict(protocol='jaxnav-transfer-v2', arm=arm, seed=seed,
        collector_variant='fixed', gamma=.999, alpha=.001, batch_size=256,
        buffer_capacity=100000, update_every=4, target_every=1000,
        network='connect4-cnn-32-64-32-dense-256-128-leakyrelu', gradient_clip=10.,
        n_envs=256, collect_steps=42, blocks=700, eval_every_blocks=10,
        epsilon_start=1., epsilon_end=.05, exploration_fraction=.5,
        total_decisions=7526400, cf_horizon=60, cf_n_rollouts=1,
        cf_gamma=.99, cf_rollout_temperature=0., cf_action_prob_temperature=.5,
        cf_score_chunk=256, consequence_metric='wasserstein',
        consequence_aggregation='weighted_mean', refresh=False,
        decay_start=100000, decay_end=150000, decay_enabled=arm == 'cce_decay',
        score_on_insert=arm == 'cce_decay', normalise_priorities='rank' if arm == 'cce_decay' else 'none',
        priority_mixing='multiplicative', mu_c=1., mu_delta=1.,
        replay_eps=.01, replay_beta=.25, maximum_priority=1.,
        eval_bank_hash=bank_hash, training_bank_hash=digest(TRAIN_BANK), smoke=False)


def validate(config):
    expected = config_for(config['arm'], config['seed'])
    if config.get('smoke'):
        expected.update(smoke_overrides(expected))
    if config != expected:
        raise ValueError('Unmatched or unknown v2 config settings')
    return config


def smoke_overrides(config):
    # Full model, full test bank, full buffer; two full-size blocks time production.
    return dict(blocks=2, eval_every_blocks=1, smoke=True)
