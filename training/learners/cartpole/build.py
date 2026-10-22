"""Registry adapter for the CartPole replay-sampling benchmark."""

from copy import deepcopy
import math
from pathlib import Path

from .config import DEFAULT_CONFIG


def validate_config(config):
    unknown = set(config) - set(DEFAULT_CONFIG)
    missing = set(DEFAULT_CONFIG) - set(config)
    if unknown or missing:
        raise ValueError(f"Unknown settings: {sorted(unknown)}; unresolved: {sorted(missing)}")
    if config["algorithm"] != "ddqn" or config["replay"] not in ("uniform", "paper_per", "repo_per"):
        raise ValueError("CartPole requires DDQN and a declared replay arm")
    for key in ("seed", "total_steps", "learning_starts", "buffer_capacity", "batch_size",
                "train_frequency", "gradient_steps", "target_update_frequency", "hidden_size",
                "eval_frequency", "eval_episodes"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a positive integer")
    if config["seed"] >= 2**32:
        raise ValueError("seed must fit uint32")
    if config["batch_size"] > config["buffer_capacity"]:
        raise ValueError("Batch exceeds replay capacity")
    if config["learning_starts"] > config["total_steps"]:
        raise ValueError("Learning starts after training ends")
    for key in ("learning_rate", "gamma", "epsilon_start", "epsilon_end", "epsilon_fraction",
                "eval_epsilon", "solve_threshold", "paper_per_alpha", "paper_per_beta_start",
                "paper_per_beta_end", "paper_per_epsilon", "repo_per_exponent",
                "repo_per_epsilon", "max_grad_norm"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["paper_per_beta_start"] > config["paper_per_beta_end"]:
        raise ValueError("PER beta schedule runs backward")
    return config


def build_agent(config):
    validate_config(config)
    from .dqn import CartPoleDDQN
    return CartPoleDDQN(deepcopy(config))


def smoke_overrides(config):
    return {
        "total_steps": 1_024,
        "learning_starts": 128,
        "buffer_capacity": 2_048,
        "batch_size": 32,
        "train_frequency": 128,
        "gradient_steps": 2,
        "eval_frequency": 512,
        "eval_episodes": 2,
        "solve_threshold": 501.0,
    }


def evaluation_axis(config):
    return "env_steps"


def simulator_sources(config):
    from gymnasium.envs.classic_control import cartpole
    return {"gymnasium/cartpole.py": Path(cartpole.__file__).resolve()}
