# Migration Guide: Old to New Structure

This guide helps you update code that was written for the old `counterfactual_frozenlake/counterfactual_rl/` structure to work with the new professional package layout.

## What Changed?

### Old Structure
```
counterfactual_frozenlake/
├── counterfactual_rl/       ← Nested inside counterfactual_frozenlake
│   ├── analysis/
│   ├── environments/
│   ├── visualization/
│   └── utils/
└── (other files)
```

### New Structure
```
counterfactual-reasoning/
├── src/
│   └── counterfactual_rl/   ← Professional src/ layout
│       ├── analysis/
│       ├── environments/
│       ├── visualization/
│       └── utils/
├── examples/
├── tests/
└── docs/
```

## Import Path Changes

### Analysis Module

**OLD:**
```python
from counterfactual_rl.analysis.counterfactual import CounterfactualAnalyzer
from counterfactual_rl.analysis.metrics import calculate_metrics
```

**NEW:**
```python
from counterfactual_rl.analysis.counterfactual import CounterfactualAnalyzer
from counterfactual_rl.analysis.metrics import calculate_metrics
```

✅ **No change needed** - same import paths! The `src/` layout is transparent to users.

### Environments Module

**OLD:**
```python
from counterfactual_rl.environments.state_manager import StateManager
from counterfactual_rl.environments.frozenlake_manager import FrozenLakeStateManager
```

**NEW:**
```python
from counterfactual_rl.environments.state_manager import StateManager
from counterfactual_rl.environments.frozenlake_manager import FrozenLakeStateManager
from counterfactual_rl.environments.registry import get_environment_manager, register_environment
```

✅ **No breaking changes** - new registry system is additive.

### Visualization Module

**OLD:**
```python
from counterfactual_rl.visualization.visualization import CounterfactualVisualizer
```

**NEW:**
```python
from counterfactual_rl.visualization.visualization import CounterfactualVisualizer
from counterfactual_rl.visualization.registry import get_visualizer
```

✅ **No breaking changes** - new registry is optional.

## API Changes

### CounterfactualAnalyzer: Now Environment-Agnostic

**OLD (FrozenLake-specific):**
```python
from counterfactual_rl.analysis.counterfactual import CounterfactualAnalyzer
from counterfactual_rl.environments.frozenlake_manager import FrozenLakeStateManager

env = gym.make("FrozenLake-v1")
state_manager = FrozenLakeStateManager(env)
analyzer = CounterfactualAnalyzer(state_manager)  # Had to use specific manager
```

**NEW (Works with any environment):**
```python
from counterfactual_rl.analysis.counterfactual import CounterfactualAnalyzer
from counterfactual_rl.environments.registry import get_environment_manager

# Use with FrozenLake
manager = get_environment_manager("frozenlake")
state_manager = manager.create_state_manager(env)
analyzer = CounterfactualAnalyzer(state_manager)

# Use with Taxi (same code!)
manager = get_environment_manager("taxi")
state_manager = manager.create_state_manager(env)
analyzer = CounterfactualAnalyzer(state_manager)  # No code changes needed
```

### Key Improvement
`CounterfactualAnalyzer` no longer has hardcoded FrozenLake logic. It works through the `StateManager` interface, making it truly environment-agnostic.

**What changed in the implementation:**
```python
# OLD: analyzer.py had hardcoded logic like this
grid_size = 4  # Hardcoded!
state_grid = state.reshape(grid_size, grid_size)

# NEW: analyzer.py uses delegation
state_info = self.state_manager.get_state_info(state)
# Each StateManager implements get_state_info() for their environment
```

## Step-by-Step Migration

### Step 1: Update Installation

**OLD:**
```bash
cd counterfactual_frozenlake
# No formal installation
```

**NEW:**
```bash
cd counterfactual-reasoning
pip install -e .  # Install development mode
# Or: pip install -e ".[dev]"  # With development dependencies
```

### Step 2: Update Import Statements (Usually No Changes Needed)

Check your imports. Most should work as-is:

```python
# These still work exactly the same:
from counterfactual_rl.analysis import CounterfactualAnalyzer
from counterfactual_rl.environments.state_manager import StateManager
from counterfactual_rl.visualization import CounterfactualVisualizer
```

### Step 3: Update Environment Setup (Optional)

If you want to use the new registry system:

**OLD:**
```python
from counterfactual_rl.environments.frozenlake_manager import FrozenLakeStateManager
state_manager = FrozenLakeStateManager(env)
```

**NEW (Better):**
```python
from counterfactual_rl.environments.registry import get_environment_manager
manager = get_environment_manager("frozenlake")
state_manager = manager.create_state_manager(env)
```

Benefits:
- Easier to switch environments
- More uniform API
- Extensible to custom environments

### Step 4: Update File Locations (If You Created Custom Code)

**OLD Structure:**
```
counterfactual_frozenlake/
├── counterfactual_rl/
│   └── environments/
│       └── my_env_manager.py  ← Custom environment
└── analysis_script.py          ← Your script
```

**NEW Structure:**
```
counterfactual-reasoning/
├── src/counterfactual_rl/
│   └── environments/
│       └── my_env_manager.py  ← Custom environment (moved here)
└── examples/
    └── analysis_script.py      ← Your script (moved here)
```

**Update your script imports:**
```python
# OLD
import sys
sys.path.insert(0, '..')
from counterfactual_rl.analysis import CounterfactualAnalyzer

# NEW (works after `pip install -e .`)
from counterfactual_rl.analysis import CounterfactualAnalyzer
```

### Step 5: Update Tests (If You Have Them)

**OLD:**
```python
# test_my_code.py
import sys
sys.path.insert(0, '../counterfactual_rl')
from analysis import MyAnalysis
```

**NEW:**
```python
# tests/test_my_code.py
from counterfactual_rl.analysis import MyAnalysis
```

Run with:
```bash
cd counterfactual-reasoning
pytest  # Works automatically from root
```

## Adding Custom Environments

### Register Your StateManager

In your code or in `src/counterfactual_rl/environments/__init__.py`:

```python
from counterfactual_rl.environments.registry import register_environment
from counterfactual_rl.environments.state_manager import StateManager

class MyCustomEnvStateManager(StateManager):
    def get_state_info(self, state):
        # Parse state for your environment
        return {
            "state": state,
            "position": state // 10,
            "features": [...]
        }

# Register it
register_environment("my_custom_env", MyCustomEnvStateManager)

# Now use it anywhere:
from counterfactual_rl.environments.registry import get_environment_manager
manager = get_environment_manager("my_custom_env")
state_manager = manager.create_state_manager(env)
```

## Common Problems and Solutions

### Problem: "ModuleNotFoundError: No module named 'counterfactual_rl'"

**Cause:** Didn't install the package.

**Solution:**
```bash
cd counterfactual-reasoning
pip install -e .
```

### Problem: "ImportError: cannot import name 'SomeClass' from 'counterfactual_rl'"

**Cause:** Class was moved or renamed.

**Solution:** Check `src/counterfactual_rl/` for where it's located now. Most classes are in the same place.

### Problem: Scripts work in IDE but fail in terminal

**Cause:** Running from wrong directory.

**Solution:** Always run from project root:
```bash
cd counterfactual-reasoning
python examples/my_script.py
```

### Problem: "Cannot find module" when importing in custom script

**Cause:** Script location doesn't matter anymore - use absolute imports.

**Solution:**
```python
# Works from ANYWHERE if package is installed:
from counterfactual_rl.analysis import CounterfactualAnalyzer

# Not this:
import sys
sys.path.insert(0, '..')
```

## Summary Table

| Aspect | Old | New | Action |
|--------|-----|-----|--------|
| **Installation** | Copy files | `pip install -e .` | Required |
| **Imports** | `from counterfactual_rl...` | `from counterfactual_rl...` | No change |
| **Environment setup** | Direct `StateManager` | Registry system optional | Optional upgrade |
| **Analyzer | FrozenLake-specific | Environment-agnostic | Automatic |
| **Script location | Root folder | `examples/` | Recommended move |
| **Test location | Scattered | `tests/` | Organized |
| **Documentation | Various files | `docs/` | Centralized |

## Checklist for Migration

- [ ] Run `pip install -e .` in project root
- [ ] Update any absolute path imports to use `from counterfactual_rl...`
- [ ] Remove any `sys.path.insert(0, ...)` hacks
- [ ] Move your analysis scripts to `examples/`
- [ ] Move your tests to `tests/`
- [ ] Update `CounterfactualAnalyzer` usage (should work as-is)
- [ ] Test scripts run without errors
- [ ] Consider using registry system for new code

## Getting Help

- See [STRUCTURE.md](STRUCTURE.md) for where things are located
- See [REGISTRY_GUIDE.md](docs/REGISTRY_GUIDE.md) for registry system details
- See [README.md](README.md) for overview
- Check [examples/](examples/) for working code samples

Welcome to the new structure! 🎉
