"""
Unit tests for registry system
"""

import pytest
from counterfactual_rl.environments import registry as env_registry
from counterfactual_rl.visualization import registry as viz_registry
from counterfactual_rl.environments import StateManager, EnvironmentConfig
from counterfactual_rl.visualization.base import EnvironmentVisualizer


# ============================================================================
# Test Fixtures and Mock Classes
# ============================================================================

class MockStateManager(StateManager):
    """Mock state manager for testing."""
    
    @staticmethod
    def clone_state(env):
        return {"test": "state"}
    
    @staticmethod
    def restore_state(env, state_dict):
        pass
    
    @staticmethod
    def get_state_info(env):
        return {"test": "info"}


class MockConfig(EnvironmentConfig):
    """Mock config for testing."""
    
    @property
    def observation_space_size(self):
        return 10
    
    @property
    def action_space_size(self):
        return 4


class MockVisualizer(EnvironmentVisualizer):
    """Mock visualizer for testing."""
    
    def __init__(self, test_param=None):
        self.test_param = test_param
    
    def plot_grid(self, state_dict=None):
        pass
    
    def plot_action_consequences(self, state, actions):
        pass


# ============================================================================
# Tests for Environment Registry
# ============================================================================

class TestEnvironmentRegistry:
    """Test environment registry functionality."""
    
    def test_register_environment(self):
        """Test registering a new environment."""
        # Clean up if exists
        if env_registry.is_registered("TestEnv-v0"):
            # Can't unregister, so skip if already exists
            pytest.skip("TestEnv-v0 already registered")
        
        # Register
        env_registry.register("TestEnv-v0", MockStateManager, MockConfig)
        
        # Verify
        assert env_registry.is_registered("TestEnv-v0")
        assert "TestEnv-v0" in env_registry.list_registered()
    
    def test_get_state_manager(self):
        """Test getting a state manager."""
        # FrozenLake should be pre-registered
        state_mgr = env_registry.get_state_manager("FrozenLake-v1")
        assert state_mgr is not None
        assert hasattr(state_mgr, 'clone_state')
        assert hasattr(state_mgr, 'restore_state')
    
    def test_get_config(self):
        """Test getting a config."""
        # FrozenLake should be pre-registered
        config = env_registry.get_config("FrozenLake-v1")
        assert config is not None
        assert hasattr(config, 'observation_space_size')
        assert hasattr(config, 'action_space_size')
    
    def test_get_state_manager_with_kwargs(self):
        """Test getting state manager with keyword arguments."""
        state_mgr = env_registry.get_state_manager("FrozenLake-v1")
        assert state_mgr is not None
    
    def test_get_config_with_kwargs(self):
        """Test getting config with keyword arguments."""
        config = env_registry.get_config("FrozenLake-v1", slippery=True)
        assert config is not None
        assert config.slippery is True
    
    def test_list_registered_environments(self):
        """Test listing registered environments."""
        registered = env_registry.list_registered()
        assert isinstance(registered, list)
        assert len(registered) > 0
        assert "FrozenLake-v1" in registered
    
    def test_is_registered(self):
        """Test checking if environment is registered."""
        assert env_registry.is_registered("FrozenLake-v1")
        assert not env_registry.is_registered("NonExistent-v0")
    
    def test_error_on_duplicate_registration(self):
        """Test that registering same environment twice raises error."""
        with pytest.raises(ValueError, match="already registered"):
            env_registry.register("FrozenLake-v1", MockStateManager, MockConfig)
    
    def test_error_on_get_unregistered(self):
        """Test that getting unregistered environment raises error."""
        with pytest.raises(KeyError, match="No state manager registered"):
            env_registry.get_state_manager("CompletelyFake-v99")
    
    def test_error_on_get_config_unregistered(self):
        """Test that getting unregistered config raises error."""
        with pytest.raises(KeyError, match="No configuration"):
            env_registry.get_config("CompletelyFake-v99")
    
    def test_frozen_lake_auto_registration(self):
        """Test that FrozenLake is auto-registered on import."""
        assert env_registry.is_registered("FrozenLake-v1")
        assert env_registry.is_registered("FrozenLake8x8-v1")


# ============================================================================
# Tests for Visualization Registry
# ============================================================================

class TestVisualizationRegistry:
    """Test visualization registry functionality."""
    
    def test_register_visualizer(self):
        """Test registering a visualizer."""
        # Clean up if exists
        if viz_registry.is_registered("TestViz-v0"):
            pytest.skip("TestViz-v0 already registered")
        
        # Register
        viz_registry.register("TestViz-v0", MockVisualizer)
        
        # Verify
        assert viz_registry.is_registered("TestViz-v0")
        assert "TestViz-v0" in viz_registry.list_registered()
    
    def test_get_visualizer(self):
        """Test getting a visualizer."""
        viz_registry.register("TestViz-v1", MockVisualizer)
        visualizer = viz_registry.get_visualizer("TestViz-v1")
        
        assert visualizer is not None
        assert isinstance(visualizer, MockVisualizer)
    
    def test_get_visualizer_with_kwargs(self):
        """Test getting visualizer with keyword arguments."""
        viz_registry.register("TestViz-v2", MockVisualizer)
        visualizer = viz_registry.get_visualizer("TestViz-v2", test_param="hello")
        
        assert visualizer is not None
        assert visualizer.test_param == "hello"
    
    def test_list_registered_visualizers(self):
        """Test listing registered visualizers."""
        registered = viz_registry.list_registered()
        assert isinstance(registered, list)
    
    def test_is_registered_visualizer(self):
        """Test checking if visualizer is registered."""
        # Register one for testing
        viz_registry.register("TestViz-v3", MockVisualizer)
        
        assert viz_registry.is_registered("TestViz-v3")
        assert not viz_registry.is_registered("NonExistent-viz")
    
    def test_error_on_duplicate_viz_registration(self):
        """Test that registering same visualizer twice raises error."""
        viz_registry.register("TestViz-v4", MockVisualizer)
        
        with pytest.raises(ValueError, match="already registered"):
            viz_registry.register("TestViz-v4", MockVisualizer)
    
    def test_error_on_get_unregistered_visualizer(self):
        """Test that getting unregistered visualizer raises error."""
        with pytest.raises(KeyError, match="No visualizer"):
            viz_registry.get_visualizer("CompletelyFake-viz")


# ============================================================================
# Integration Tests
# ============================================================================

class TestRegistryIntegration:
    """Integration tests for the registry system."""
    
    def test_register_both_registries(self):
        """Test registering same environment to both registries."""
        # These should not interfere with each other
        env_registry.register("Integration-v0", MockStateManager, MockConfig)
        
        # Should be able to get from environment registry
        state_mgr = env_registry.get_state_manager("Integration-v0")
        assert state_mgr is not None
        
        # Visualization registry should be separate
        assert not viz_registry.is_registered("Integration-v0")
    
    def test_frozen_lake_complete_workflow(self):
        """Test complete workflow with FrozenLake."""
        # Get all components
        state_mgr = env_registry.get_state_manager("FrozenLake-v1")
        config = env_registry.get_config("FrozenLake-v1")
        
        # Verify they work
        assert state_mgr is not None
        assert config is not None
        assert config.observation_space_size == 16
        assert config.action_space_size == 4


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    # Run with: pytest tests/test_registry.py -v
    pytest.main([__file__, "-v"])
