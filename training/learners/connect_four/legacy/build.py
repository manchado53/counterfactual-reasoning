"""Historical chunk-budget learner, with original opponents and evaluator."""
import math
from .config import DEFAULT_CONNECT4_CONFIG

ENV_INFO = dict(obs_dim=84,num_agents=1,actions_per_agent=7,
                agent_names=['player0'],obs_type='flat',env_name='connect_four')


def validate_config(config):
    keys = set(DEFAULT_CONNECT4_CONFIG) | {'protocol','collector_variant','training_budget_kind',
        'early_stop_win_rate','diagnostics_n_step_slices','diagnostics_n_scatter_snapshots'}
    if set(config) != keys:
        raise ValueError('Unknown or unresolved historical ConnectFour settings')
    if (config['collector_variant'] != 'old-chunk' or config['training_budget_kind'] != 'chunks'
            or config['env_name'] != 'connect_four' or config['early_stop_win_rate'] is not None):
        raise ValueError('Historical collection/budget mismatch')
    if config['algorithm'] not in ('dqn-uniform','dqn','consequence-dqn'):
        raise ValueError('Unknown historical learner')
    if config['opponent'] not in ('random','rule_based','mcts'):
        raise ValueError('Unknown historical opponent')
    if config['eval_opponent'] != 'random':
        raise ValueError('Source eval_opponent is inert; evaluator actually uses opponent')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    for key in ('n_envs','collect_steps','n_chunks','M','B','C','n_steps_for_Q_update',
                'hidden_dim','mcts_n_sims','eval_episodes','eval_interval','save_every',
                'score_interval','n_score_sample','cf_horizon','cf_n_rollouts','cf_top_k',
                'diagnostics_plot_interval','diagnostics_n_step_slices','diagnostics_n_scatter_snapshots'):
        if type(config[key]) is not int or config[key]<1:
            raise ValueError(f'Invalid {key}')
    if config['B']>config['M'] or config['cf_top_k']>7 or config['n_envs']*config['collect_steps']>config['M']:
        raise ValueError('Replay or action count exceeds bounds')
    for key in ('gamma','cf_gamma','mu','epsilon_start','epsilon_end','exploration_fraction'):
        if type(config[key]) not in (int,float) or not 0<=config[key]<=1:
            raise ValueError(f'Invalid {key}')
    for key in ('alpha','mu_c','mu_delta'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or config[key]<0:
            raise ValueError(f'Invalid {key}')
    if config['priority_mixing'] not in ('additive','multiplicative'):
        raise ValueError('Invalid replay mixing')
    if type(config['n_checkpoints']) is not int or config['n_checkpoints']<0:
        raise ValueError('Invalid checkpoint count')
    if set(config['PER_parameters']) != {'eps','beta','maximum_priority'}:
        raise ValueError('Unknown replay settings')
    return config


def build_agent(config):
    validate_config(config)
    from .runner import DQN, CCE
    cls = CCE if config['algorithm']=='consequence-dqn' else DQN
    agent = cls(ENV_INFO.copy(),config)
    agent.evaluation_identity = lambda: dict(eval_bank_hash=None,eval_fixed_bank=False,
        eval_seed_rule='zero-based chunk index during learn; evaluate defaults to 42',
        eval_opponent=config['opponent'],eval_opponent_config_inert=config['eval_opponent'],
        eval_rng_mode='independent key derived from evaluation seed',eval_max_steps=84,
        training_numpy_rng='inherited-process-state-not-seeded-by-source',
        source_log_axis='chunk',eval_axis='env_steps')
    return agent
