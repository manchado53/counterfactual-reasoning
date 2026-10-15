"""V2 replay/training controls with the approved fixed minimax test bank."""
from .protocol_v2 import config_for as v2_config, ARMS, smoke_overrides
from .bank_v3 import load


def config_for(arm, seed):
    config = v2_config(arm,seed)
    config.update(protocol='fixed-eval-v3',eval_bank_hash=load()[1])
    return config


def validate(config):
    expected = config_for(config['arm'],config['seed'])
    if config.get('smoke'):
        expected.update(smoke_overrides(expected))
    if config != expected:
        raise ValueError('Unmatched or unknown v3 config settings')
    return config
