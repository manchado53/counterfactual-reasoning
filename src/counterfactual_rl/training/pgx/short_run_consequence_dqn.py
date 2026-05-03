"""Short ConsequenceDQN training run (~7 min) with game recording enabled."""
import sys
sys.path.insert(0, __file__.split('src/')[0] + 'src')

from counterfactual_rl.training.pgx.dqn_jax.train import create_chess_env
from counterfactual_rl.training.pgx.dqn_jax.consequence_dqn import ChessConsequenceDQN

env, key, info = create_chess_env(seed=0)
agent = ChessConsequenceDQN(env, info, {
    'n_episodes':      2,
    'n_envs':          64,
    'collect_steps':   256,
    'M':               10000,
    'B':               32,
    'eval_interval':   None,
    'record_interval': 1,    # records a game after every chunk
    'score_interval':  1,
    'n_score_sample':  16,
    'cf_n_rollouts':   4,
    'cf_horizon':      3,
    'cf_top_k':        5,
    'save_every':      9999,
})
agent.learn()
print("Short ConsequenceDQN run complete")
