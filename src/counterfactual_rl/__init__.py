"""
Counterfactual Reinforcement Learning

A framework for analyzing consequential states in RL using counterfactual reasoning.
"""

__version__ = "0.1.0"
__author__ = "RL Research Team"

from counterfactual_rl.utils.data_structures import ConsequenceRecord
from counterfactual_rl.environments.frozen_lake import FrozenLakeStateManager

# Training module
from counterfactual_rl import training

__all__ = [
    "ConsequenceRecord",
    "FrozenLakeStateManager",
    "training",
]
