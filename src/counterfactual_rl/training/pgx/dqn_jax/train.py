"""
Entry point for Gardner chess DQN training.

Usage (direct):
    python -m counterfactual_rl.training.pgx.dqn_jax.train

Usage (with config overrides):
    CONFIG_OVERRIDES='{"n_episodes": 5000, "algorithm": "dqn"}' \\
        python -m counterfactual_rl.training.pgx.dqn_jax.train

    CONFIG_OVERRIDES_B64=<base64-encoded-json> \\
        python -m counterfactual_rl.training.pgx.dqn_jax.train

Config override keys (any key from DEFAULT_CHESS_CONFIG):
    algorithm:   'consequence-dqn' (default) | 'dqn' | 'dqn-uniform'
    n_episodes:  training episodes
    seed:        RNG seed
    ... (see config.py for full list)
"""

import base64
import json
import os

import jax

from .chess_env import GardnerChessEnv
from .config import DEFAULT_CHESS_CONFIG


def create_chess_env(seed: int = 0, opponent: str = 'random'):
    """
    Instantiate GardnerChessEnv and return (env, jax_key, env_info).

    Args:
        seed:     RNG seed for the environment and JAX key.
        opponent: 'random' (default) or 'baseline'.

    Returns:
        env:      GardnerChessEnv instance
        key:      jax.random.PRNGKey(seed)
        env_info: dict with obs_dim, num_agents, actions_per_agent, etc.
    """
    env = GardnerChessEnv(seed=seed, opponent=opponent)
    env_info = {
        'obs_dim':          2875,
        'num_agents':       1,
        'actions_per_agent': 1225,
        'agent_names':      ['white'],
        'obs_type':         'flat',
        'env_name':         'gardner_chess',
    }
    return env, jax.random.PRNGKey(seed), env_info


def main():
    config = DEFAULT_CHESS_CONFIG.copy()

    # Load config overrides from environment variables (same pattern as SMAX)
    raw_b64 = os.environ.get('CONFIG_OVERRIDES_B64', '')
    raw_json = os.environ.get('CONFIG_OVERRIDES', '')

    if raw_b64:
        overrides = json.loads(base64.b64decode(raw_b64).decode())
        config.update(overrides)
    elif raw_json:
        overrides = json.loads(raw_json)
        config.update(overrides)

    seed = config.get('seed', 0)
    opponent = config.get('opponent', 'random')

    env, _key, env_info = create_chess_env(seed=seed, opponent=opponent)

    algorithm = config.get('algorithm', 'consequence-dqn')
    if algorithm == 'consequence-dqn':
        from .consequence_dqn import ChessConsequenceDQN
        agent = ChessConsequenceDQN(env, env_info, config=config)
    else:
        from .dqn import ChessDQN
        agent = ChessDQN(env, env_info, config=config)

    agent.learn()


if __name__ == '__main__':
    main()
