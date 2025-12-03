# Registry System Guide

The registry system is a simple, pythonic way to register and manage environment-specific components (state managers, configs, and visualizers) without complex factory patterns.

## Quick Start

### Using an Existing Environment

```python
from counterfactual_rl.environments import registry as env_registry
from counterfactual_rl.visualization import registry as viz_registry

# Get state manager and config for FrozenLake
state_manager = env_registry.get_state_manager("FrozenLake-v1")
config = env_registry.get_config("FrozenLake-v1")

# Get visualizer for FrozenLake
visualizer = viz_registry.get_visualizer("FrozenLake-v1")

# List all registered components
print(env_registry.list_registered())  # ['FrozenLake-v1', 'FrozenLake8x8-v1']
```

## API Reference

### Environment Registry (`counterfactual_rl.environments.registry`)

#### `register(env_id, StateManagerClass, ConfigClass)`
Register a new environment with its state manager and config.

```python
from counterfactual_rl.environments import registry
from my_environments import MyStateManager, MyConfig

registry.register("MyEnv-v1", MyStateManager, MyConfig)
```

**Parameters:**
- `env_id` (str): Unique environment identifier
- `StateManagerClass` (class): Subclass of `StateManager`
- `ConfigClass` (class): Subclass of `EnvironmentConfig`

**Raises:**
- `ValueError` if environment already registered

---

#### `get_state_manager(env_id, *args, **kwargs)`
Get an instantiated state manager for an environment.

```python
state_manager = registry.get_state_manager("FrozenLake-v1")
```

**Parameters:**
- `env_id` (str): Environment identifier
- `*args, **kwargs`: Arguments passed to state manager constructor

**Returns:** Instance of registered state manager

**Raises:**
- `KeyError` if environment not registered

---

#### `get_config(env_id, **kwargs)`
Get an instantiated config for an environment.

```python
config = registry.get_config("FrozenLake-v1", slippery=True)
```

**Parameters:**
- `env_id` (str): Environment identifier
- `**kwargs`: Arguments passed to config constructor

**Returns:** Instance of registered config

**Raises:**
- `KeyError` if environment not registered

---

#### `list_registered()`
List all registered environment IDs.

```python
envs = registry.list_registered()
# Returns: ['FrozenLake-v1', 'FrozenLake8x8-v1']
```

**Returns:** List of environment IDs

---

#### `is_registered(env_id)`
Check if an environment is registered.

```python
if registry.is_registered("FrozenLake-v1"):
    print("FrozenLake is registered!")
```

**Parameters:**
- `env_id` (str): Environment identifier

**Returns:** Boolean

---

### Visualization Registry (`counterfactual_rl.visualization.registry`)

#### `register(env_id, VisualizerClass)`
Register a visualizer for an environment.

```python
from counterfactual_rl.visualization import registry
from my_visualizers import MyVisualizer

registry.register("MyEnv-v1", MyVisualizer)
```

**Parameters:**
- `env_id` (str): Environment identifier
- `VisualizerClass` (class): Visualizer class

**Raises:**
- `ValueError` if visualizer already registered

---

#### `get_visualizer(env_id, *args, **kwargs)`
Get an instantiated visualizer for an environment.

```python
visualizer = registry.get_visualizer("FrozenLake-v1")
```

**Parameters:**
- `env_id` (str): Environment identifier
- `*args, **kwargs`: Arguments passed to visualizer constructor

**Returns:** Instance of registered visualizer

**Raises:**
- `KeyError` if visualizer not registered

---

#### `list_registered()` and `is_registered(env_id)`
Same as environment registry.

---

## Adding a New Environment

### Step 1: Create State Manager

```python
# my_environments/taxi_state_manager.py
from counterfactual_rl.environments import StateManager, EnvironmentConfig

class TaxiStateManager(StateManager):
    @staticmethod
    def clone_state(env):
        return {
            'taxi_row': env.unwrapped.taxi_row,
            'taxi_col': env.unwrapped.taxi_col,
            'passenger_location': env.unwrapped.passenger_location,
            'destination': env.unwrapped.destination,
            'np_random_state': copy.deepcopy(env.unwrapped.np_random.bit_generator.state),
        }
    
    @staticmethod
    def restore_state(env, state_dict):
        env.unwrapped.taxi_row = state_dict['taxi_row']
        env.unwrapped.taxi_col = state_dict['taxi_col']
        env.unwrapped.passenger_location = state_dict['passenger_location']
        env.unwrapped.destination = state_dict['destination']
        env.unwrapped.np_random.bit_generator.state = state_dict['np_random_state']
    
    @staticmethod
    def get_state_info(env):
        return {
            'taxi_position': (env.unwrapped.taxi_row, env.unwrapped.taxi_col),
            'passenger_location': env.unwrapped.passenger_location,
            'destination': env.unwrapped.destination,
        }
```

### Step 2: Create Config

```python
# my_environments/taxi_config.py
from counterfactual_rl.environments import EnvironmentConfig

class TaxiConfig(EnvironmentConfig):
    def __init__(self):
        pass
    
    @property
    def observation_space_size(self):
        return 500  # 5x5 grid * 5 passenger locations * 4 destinations
    
    @property
    def action_space_size(self):
        return 6  # left, down, right, up, pickup, dropoff
    
    @property
    def grid_shape(self):
        return (5, 5)
```

### Step 3: Create Visualizer

```python
# my_visualizers/taxi_visualizer.py
from counterfactual_rl.visualization.base import EnvironmentVisualizer

class TaxiVisualizer(EnvironmentVisualizer):
    def __init__(self, env=None):
        self.env = env
    
    def plot_grid(self, state_dict=None):
        # Your visualization logic here
        pass
    
    def plot_action_consequences(self, state, actions):
        # Your visualization logic here
        pass
```

### Step 4: Register Everything

```python
# my_environments/__init__.py
from counterfactual_rl.environments import registry as env_registry
from counterfactual_rl.visualization import registry as viz_registry
from my_environments.taxi_state_manager import TaxiStateManager, TaxiConfig
from my_visualizers.taxi_visualizer import TaxiVisualizer

# Auto-register on module import
env_registry.register("Taxi-v3", TaxiStateManager, TaxiConfig)
viz_registry.register("Taxi-v3", TaxiVisualizer)
```

### Step 5: Use It

```python
from my_environments import *  # Triggers auto-registration
from counterfactual_rl.environments import registry as env_registry
from counterfactual_rl.visualization import registry as viz_registry

# Now Taxi is registered
state_mgr = env_registry.get_state_manager("Taxi-v3")
visualizer = viz_registry.get_visualizer("Taxi-v3")
```

## Migration from Factory Pattern

### Old Way (Factory Pattern)

```python
from counterfactual_rl.environments import StateManagerFactory

state_manager = StateManagerFactory.get_state_manager("FrozenLake-v1")
config = StateManagerFactory.get_config("FrozenLake-v1")
StateManagerFactory.register("NewEnv-v1", MyStateManager, MyConfig)
```

### New Way (Registry Pattern)

```python
from counterfactual_rl.environments import registry

state_manager = registry.get_state_manager("FrozenLake-v1")
config = registry.get_config("FrozenLake-v1")
registry.register("NewEnv-v1", MyStateManager, MyConfig)
```

**Benefits:**
- ✅ 50% less code
- ✅ More Pythonic (dict-based)
- ✅ Easier to extend
- ✅ Better readability

## Understanding Auto-Registration

When you import `counterfactual_rl.environments`, the `__init__.py` file automatically registers FrozenLake:

```python
# counterfactual_rl/environments/__init__.py
from counterfactual_rl.environments import registry

# Auto-register FrozenLake on module import
registry.register("FrozenLake-v1", FrozenLakeStateManager, FrozenLakeConfig)
```

This means you can immediately use FrozenLake without manual registration:

```python
import counterfactual_rl  # Registration happens automatically
from counterfactual_rl.environments import registry

state_mgr = registry.get_state_manager("FrozenLake-v1")  # Works!
```

## Troubleshooting

### KeyError: "No state manager registered for 'MyEnv-v1'"

**Problem:** Environment not registered

**Solution:** Make sure you imported the module that registers it:
```python
from my_environments import *  # This triggers auto-registration
```

### ValueError: "State manager for 'FrozenLake-v1' already registered"

**Problem:** Tried to register same environment twice

**Solution:** Only register once during module initialization. Put registration in `__init__.py`, not in user code.

### How do I check what's registered?

```python
from counterfactual_rl.environments import registry
print(registry.list_registered())  # See all registered environments
print(registry.is_registered("FrozenLake-v1"))  # Check specific environment
```

## Design Principles

1. **Simple**: Dict-based, no complex factory classes
2. **Pythonic**: Module-level registration, auto-registration on import
3. **Extensible**: Easy to add new environments
4. **Type-safe**: Use abstract base classes for interface enforcement
5. **Backward-compatible**: Old factory code still works for reference

## See Also

- `examples/adding_taxi_environment.py` - Complete example of adding Taxi-v3
- `tests/test_registry.py` - Unit tests for registry system
- `MIGRATION_CHECKLIST.md` - Step-by-step migration guide
