"""
Smoke test for vectorized ChessDQN.
Run directly or submit via test_chess.sh.
"""
import sys
sys.path.insert(0, __file__.split('src/')[0] + 'src')

from counterfactual_rl.training.pgx.dqn_jax.train import create_chess_env
from counterfactual_rl.training.pgx.dqn_jax.dqn import ChessDQN

env, key, info = create_chess_env(seed=0)
agent = ChessDQN(env, info, {
    'n_episodes':    50,
    'n_envs':        64,
    'collect_steps': 256,
    'M':             10000,
    'B':             32,
    'eval_interval': 25,
    'eval_episodes': 10,
    'algorithm':     'dqn',
    'save_every':    9999,
})
agent.learn()
metrics = agent.evaluate(n_episodes=10)
assert 'win_rate' in metrics
print(f"ChessDQN vectorized OK | metrics: {metrics}")
