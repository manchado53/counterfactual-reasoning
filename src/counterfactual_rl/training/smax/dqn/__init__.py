"""DQN implementation for SMAX (modular, SB3-style)."""

from .policies import CentralizedQNetwork
from .buffers import PrioritizedReplayBuffer
from .config import DEFAULT_CONFIG
from .utils import (
    create_smax_env,
    plot_training_curves,
    get_action_masks,
    get_global_state,
    get_global_reward,
    is_done,
    record_episode,
    save_gameplay_gif,
)
from .dqn import DQN
from .train import main as train_main

__all__ = [
    # Agent
    'DQN',
    # Network
    'CentralizedQNetwork',
    # Buffer
    'PrioritizedReplayBuffer',
    # Config
    'DEFAULT_CONFIG',
    # Utils
    'create_smax_env',
    'plot_training_curves',
    'get_action_masks',
    'get_global_state',
    'get_global_reward',
    'is_done',
    'record_episode',
    'save_gameplay_gif',
    # Entry point
    'train_main',
]
