"""Entry point for training DQN on SMAX. All parameters come from config.py."""

import base64
import json
import os
from .config import DEFAULT_CONFIG, SCENARIO_PRESETS
from counterfactual_rl.envs.smax import create_smax_env
from .utils import evaluate


def main():
    """Train DQN agent on SMAX."""
    config = DEFAULT_CONFIG.copy()

    # Parse explicit overrides from environment (set by run_experiments.py)
    # Base64-encoded to avoid SLURM --export splitting on commas in JSON
    overrides = {}
    overrides_b64 = os.environ.get('CONFIG_OVERRIDES_B64')
    overrides_json = os.environ.get('CONFIG_OVERRIDES')
    if overrides_b64:
        overrides_json = base64.b64decode(overrides_b64).decode()
    if overrides_json:
        overrides = json.loads(overrides_json)
        for key in overrides:
            if key not in DEFAULT_CONFIG:
                print(f"Warning: unknown config key '{key}'")

    # Apply scenario preset (intermediate priority: defaults < preset < overrides)
    scenario = overrides.get('scenario', config['scenario'])
    if scenario in SCENARIO_PRESETS:
        config.update(SCENARIO_PRESETS[scenario])

    # Apply explicit overrides last (highest priority)
    for key, value in overrides.items():
        config[key] = value

    # Allow env var override for SLURM job arrays (e.g. metric sweeps)
    metric_override = os.environ.get('CONSEQUENCE_METRIC')
    if metric_override:
        config['consequence_metric'] = metric_override

    # Dynamic import based on algorithm
    if config['algorithm'] == 'consequence-dqn':
        from .consequence_dqn import ConsequenceDQN as DQN
    else:
        from .dqn import DQN

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
