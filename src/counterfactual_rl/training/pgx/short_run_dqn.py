"""Short DQN training run (~4 min) with game recording enabled."""
import sys
sys.path.insert(0, __file__.split('src/')[0] + 'src')

from counterfactual_rl.training.pgx.dqn_jax.train import create_chess_env
from counterfactual_rl.training.pgx.dqn_jax.dqn import ChessDQN

env, key, info = create_chess_env(seed=0)
agent = ChessDQN(env, info, {
    'n_episodes':      8,
    'n_envs':          64,
    'collect_steps':   256,
    'M':               10000,
    'B':               32,
    'eval_interval':   None,
    'record_interval': 4,    # records a game at chunk 3 and chunk 7
    'save_every':      9999,
    'algorithm':       'dqn',
})
agent.learn()
print("Short DQN run complete")
