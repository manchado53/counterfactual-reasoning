"""Matched depth-3 training and frozen depth-3 primary evaluation."""
from .protocol_v3 import config_for as previous, ARMS, smoke_overrides
from .bank_v2 import digest
from .minimax_eval import SPEC

TRAIN_BANK = dict(search=SPEC,depth=3,roles=[0,1],ties='uniform-among-exact-root-best',
                  randomness='explicit-jax-key-per-reply',implementation='jax-bitboard-depth3-v1')


def config_for(arm,seed):
    c = previous(arm,seed)
    c.update(protocol='minimax3-v4',training_bank_hash=digest(TRAIN_BANK),
        training_opponent=TRAIN_BANK,primary_eval_depth=3,
        cf_n_rollouts=4,cf_common_random_numbers=True,cf_score_chunk=64)
    return c


def validate(config):
    expected = config_for(config['arm'],config['seed'])
    if config.get('smoke'):
        expected.update(smoke_overrides(expected))
    if config != expected:
        raise ValueError('Unmatched or unknown minimax3-v4 config')
    return config
