"""
Tests for FrozenLake state manager
"""

import pytest
import gymnasium as gym
import numpy as np

from counterfactual_rl.environments.frozen_lake import FrozenLakeStateManager


class TestFrozenLakeStateManager:
    """Test suite for FrozenLake state management"""

    def test_clone_and_restore(self):
        """Test that we can clone and restore state correctly"""
        env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
        env.reset()

        manager = FrozenLakeStateManager()

        # Move to a specific state
        env.unwrapped.s = 5

        # Clone the state
        state_dict = manager.clone_state(env)

        # Change the state
        env.unwrapped.s = 10

        # Restore the state
        manager.restore_state(env, state_dict)

        # Verify restoration
        assert env.unwrapped.s == 5, "State should be restored to 5"

    def test_rng_state_restoration(self):
        """Test that RNG state is correctly restored"""
        env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
        env.reset()

        manager = FrozenLakeStateManager()

        # Save state
        state_dict = manager.clone_state(env)

        # Take an action (which uses RNG in slippery mode)
        env.step(0)

        # Restore state
        manager.restore_state(env, state_dict)

        # Take the same action twice - should give same result
        obs1, reward1, term1, trunc1, info1 = env.step(0)

        # Restore again
        manager.restore_state(env, state_dict)
        obs2, reward2, term2, trunc2, info2 = env.step(0)

        # Results should be identical
        assert obs1 == obs2, "Observations should match"
        assert reward1 == reward2, "Rewards should match"
        assert term1 == term2, "Termination should match"

    def test_multiple_clones(self):
        """Test that multiple clones are independent"""
        env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
        env.reset()

        manager = FrozenLakeStateManager()

        # Create two different states
        env.unwrapped.s = 3
        state1 = manager.clone_state(env)

        env.unwrapped.s = 7
        state2 = manager.clone_state(env)

        # Restore first state
        manager.restore_state(env, state1)
        assert env.unwrapped.s == 3

        # Restore second state
        manager.restore_state(env, state2)
        assert env.unwrapped.s == 7

        # Restore first state again
        manager.restore_state(env, state1)
        assert env.unwrapped.s == 3


def test_state_manager_factory():
    """Test the state manager factory"""
    from counterfactual_rl.environments.frozen_lake import StateManagerFactory

    # Should work for FrozenLake
    manager = StateManagerFactory.create("FrozenLake-v1")
    assert isinstance(manager, FrozenLakeStateManager)

    # Should raise error for unknown environment
    with pytest.raises(NotImplementedError):
        StateManagerFactory.create("UnknownEnv-v0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
