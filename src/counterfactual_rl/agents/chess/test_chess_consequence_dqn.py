"""
Smoke test for vectorized ChessConsequenceDQN.
Run directly or submit via test_chess.sh.
"""
import sys
sys.path.insert(0, __file__.split('src/')[0] + 'src')

from counterfactual_rl.agents.chess.train import create_chess_env
from counterfactual_rl.agents.chess.consequence_dqn import ChessConsequenceDQN

env, key, info = create_chess_env(seed=0)
agent = ChessConsequenceDQN(env, info, {
    'n_episodes':    20,
    'n_envs':        64,
    'collect_steps': 256,
    'M':             10000,
    'B':             32,
    'eval_interval': 10,
    'eval_episodes': 5,
    'score_interval':  5,
    'n_score_sample': 16,
    'cf_n_rollouts':   4,
    'cf_horizon':      3,
    'cf_top_k':        5,
    'save_every':    9999,
})
agent.learn()
print("ChessConsequenceDQN vectorized OK")
