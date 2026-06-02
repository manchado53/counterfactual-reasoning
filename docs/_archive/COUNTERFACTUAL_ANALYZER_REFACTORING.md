# CounterfactualAnalyzer - Now Fully Environment-Agnostic ✅

## What Changed

The `CounterfactualAnalyzer` class has been refactored to work with **any environment**, not just FrozenLake.

### Key Changes

#### 1. **Import Changed** (Line 12)
```python
# OLD (FrozenLake-specific)
from counterfactual_rl.environments.state_manager import FrozenLakeStateManager

# NEW (Generic)
from counterfactual_rl.environments.base import StateManager
```

#### 2. **Constructor Updated** (Line 40)
```python
# OLD (FrozenLake-specific type)
state_manager: FrozenLakeStateManager,

# NEW (Generic StateManager)
state_manager: StateManager,
```

#### 3. **Docstring Improved**
Now shows examples for both FrozenLake AND Taxi-v3:
```python
Example:
    # For FrozenLake
    from counterfactual_rl.environments import registry
    state_manager = registry.get_state_manager("FrozenLake-v1")
    analyzer = CounterfactualAnalyzer(model, env, state_manager)

    # For Taxi-v3 (once registered)
    state_manager = registry.get_state_manager("Taxi-v3")
    analyzer = CounterfactualAnalyzer(model, env, state_manager)
    # Same analyzer code works!
```

#### 4. **Hardcoded Grid Logic Removed** (Lines 293-299)
```python
# OLD (FrozenLake-hardcoded)
grid_size = 4  # TODO: Make this configurable
row = current_position // grid_size
col = current_position % grid_size
position = (row, col)

# NEW (Environment-agnostic)
state_info = self.state_manager.get_state_info(self.env)
state_value = state_info.get('state', current_position)
position = state_info.get('position', None)
```

This delegates to the **environment-specific StateManager** to handle state parsing!

---

## How to Use With Different Environments

### FrozenLake (Already Works)
```python
from counterfactual_rl.environments import registry
from counterfactual_rl.analysis import CounterfactualAnalyzer
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FrozenLake-v1", is_slippery=False)
model = PPO.load("frozenlake_model.zip", env=env)

# Create analyzer with FrozenLake state manager
analyzer = CounterfactualAnalyzer(
    model=model,
    env=env,
    state_manager=registry.get_state_manager("FrozenLake-v1"),
    horizon=20,
    n_rollouts=50
)

# Run analysis
records = analyzer.evaluate_multiple_episodes(n_episodes=20)
```

### Taxi-v3 (Once TaxiStateManager is Registered)
```python
env = gym.make("Taxi-v3")
model = PPO.load("taxi_model.zip", env=env)

# SAME ANALYZER CODE! Just different environment + state manager
analyzer = CounterfactualAnalyzer(
    model=model,
    env=env,
    state_manager=registry.get_state_manager("Taxi-v3"),  # Different state manager
    horizon=20,
    n_rollouts=50
)

records = analyzer.evaluate_multiple_episodes(n_episodes=20)
```

### CartPole-v1 (Once CartPoleStateManager is Registered)
```python
env = gym.make("CartPole-v1")
model = PPO.load("cartpole_model.zip", env=env)

# SAME ANALYZER CODE!
analyzer = CounterfactualAnalyzer(
    model=model,
    env=env,
    state_manager=registry.get_state_manager("CartPole-v1"),
    horizon=20,
    n_rollouts=50
)

records = analyzer.evaluate_multiple_episodes(n_episodes=20)
```

---

## Why This Works

The key is that **each environment-specific StateManager handles its own state parsing**:

| Environment | StateManager | get_state_info() Returns |
|-------------|--------------|---------------------------|
| FrozenLake-v1 | `FrozenLakeStateManager` | `{'state': 0-15, 'position': (row, col)}` |
| Taxi-v3 | `TaxiStateManager` | `{'taxi_position': (r, c), 'passenger_location': 0-4, ...}` |
| CartPole-v1 | `CartPoleStateManager` | `{'position': x, 'velocity': v, 'angle': θ, ...}` |

The `CounterfactualAnalyzer` doesn't care about the specific format - it just calls:
```python
state_info = self.state_manager.get_state_info(self.env)
```

And each StateManager returns whatever makes sense for **that environment**.

---

## Files Modified

✅ `counterfactual_rl/analysis/counterfactual.py`
- Line 12: Changed import from `FrozenLakeStateManager` to `StateManager`
- Line 40: Changed parameter type from `FrozenLakeStateManager` to `StateManager`
- Lines 27-35: Updated docstring with multi-environment examples
- Lines 293-299: Replaced hardcoded grid logic with `state_manager.get_state_info()`
- Lines 302-313: Updated print statements to use generic `state_value`

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Works with | FrozenLake only | Any registered environment |
| Import | `FrozenLakeStateManager` | `StateManager` (base class) |
| Hardcoded? | Yes (grid_size=4) | No (uses StateManager) |
| Extensible? | No | Yes - register new environment + StateManager |
| Code changes to add new env? | Modify analyzer | Just register StateManager |

---

## Next Steps

To add a new environment (e.g., Taxi-v3):

1. **Create TaxiStateManager** in `counterfactual_rl/environments/taxi_state_manager.py`
2. **Implement get_state_info()** to return Taxi-specific state format
3. **Register** with `registry.register("Taxi-v3", TaxiStateManager, TaxiConfig)`
4. **Use the same CounterfactualAnalyzer** - no code changes needed!

✅ The hard work is done - now you have a truly reusable analysis framework!
