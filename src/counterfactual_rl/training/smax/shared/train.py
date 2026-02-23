"""Entry point for training DQN on SMAX. All parameters come from config.py."""

import os
from .config import DEFAULT_CONFIG
from .utils import create_smax_env, evaluate


def main():
    """Train DQN agent on SMAX."""
    config = DEFAULT_CONFIG.copy()

    # Allow env var override for SLURM job arrays (e.g. metric sweeps)
    metric_override = os.environ.get('CONSEQUENCE_METRIC')
    if metric_override:
        config['consequence_metric'] = metric_override

    # Dynamic import based on algorithm and backend
    if config['algorithm'] == 'consequence-dqn':
        from ..dqn_jax.consequence_dqn import ConsequenceDQN as DQN
    elif config['backend'] == 'jax':
        from ..dqn_jax.dqn import DQN
    else:
        from ..dqn_pytorch.dqn import DQN

    # Create environment
    env, key, env_info = create_smax_env(
        scenario=config['scenario'],
        seed=config['seed'],
        obs_type=config['obs_type'],
    )

    # Create and train agent
    agent = DQN(env, env_info, config=config)
    agent.learn()

    # Post-training evaluation
    evaluate(agent, n_episodes=config['eval_episodes'],
             parallel=(config['backend'] == 'jax'))


if __name__ == '__main__':
    main()
