"""
Entry point for Connect Four DQN training.

Usage (direct):
    python -m counterfactual_rl.agents.connect_four.train

Usage (with config overrides):
    CONFIG_OVERRIDES='{"n_chunks": 20, "algorithm": "dqn"}' \\
        python -m counterfactual_rl.agents.connect_four.train

    CONFIG_OVERRIDES_B64=<base64-encoded-json> \\
        python -m counterfactual_rl.agents.connect_four.train

Config override keys (any key from DEFAULT_CONNECT4_CONFIG):
    algorithm:   'consequence-dqn' (default) | 'dqn' | 'dqn-uniform'
    n_chunks:    number of collection chunks (each = n_envs x collect_steps transitions)
    seed:        RNG seed
    ... (see config.py for full list)
"""

import base64
import json
import os

import jax

from .config import DEFAULT_CONNECT4_CONFIG


def create_connect4_env(seed: int = 0):
    """
    Return (jax_key, env_info) for Connect Four.

    No wrapper — pgx.make('connect_four') is called directly inside the agent.
    env_info matches the shape expected by Connect4DQN.__init__().
    """
    env_info = {
        'obs_dim':           84,         # 6 x 7 x 2 = 84
        'num_agents':        1,
        'actions_per_agent': 7,          # columns 0-6
        'agent_names':       ['player0'],
        'obs_type':          'flat',
        'env_name':          'connect_four',
    }
    return jax.random.PRNGKey(seed), env_info


def main():
    config = DEFAULT_CONNECT4_CONFIG.copy()

    raw_b64 = os.environ.get('CONFIG_OVERRIDES_B64', '')
    raw_json = os.environ.get('CONFIG_OVERRIDES', '')

    if raw_b64:
        overrides = json.loads(base64.b64decode(raw_b64).decode())
        config.update(overrides)
    elif raw_json:
        overrides = json.loads(raw_json)
        config.update(overrides)

    seed = config.get('seed', 0)
    _key, env_info = create_connect4_env(seed=seed)

    algorithm = config.get('algorithm', 'consequence-dqn')
    if algorithm == 'consequence-dqn':
        from .consequence_dqn import Connect4ConsequenceDQN
        agent = Connect4ConsequenceDQN(env_info, config=config)
    else:
        from .dqn import Connect4DQN
        agent = Connect4DQN(env_info, config=config)

    agent.learn()


if __name__ == '__main__':
    main()
