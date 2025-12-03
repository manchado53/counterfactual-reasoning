"""Environment management and state handling"""

from counterfactual_rl.environments.base import StateManager, EnvironmentConfig
from counterfactual_rl.environments.frozen_lake import (
    FrozenLakeStateManager,
    FrozenLakeConfig,
)
from counterfactual_rl.environments.smac import (
    SmacStateManager,
    SmacConfig,
)
from counterfactual_rl.environments import registry

# Auto-register environments on module import
registry.register("FrozenLake-v1", FrozenLakeStateManager, FrozenLakeConfig)
registry.register("FrozenLake8x8-v1", FrozenLakeStateManager, FrozenLakeConfig)
registry.register("SMAC-3m", SmacStateManager, SmacConfig)

__all__ = [
    "StateManager",
    "EnvironmentConfig",
    "FrozenLakeStateManager",
    "FrozenLakeConfig",
    "SmacStateManager",
    "SmacConfig",
    "registry",
]
