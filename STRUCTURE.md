# Repository Structure Guide

This document explains the organization of the `counterfactual-reasoning` project and why files are located where they are.

## High-Level Organization

The project follows Python packaging best practices with a **src/ layout**:

```
counterfactual-reasoning/
├── src/counterfactual_rl/     ← The actual Python package
├── examples/                   ← Usage examples and analysis scripts
├── tests/                      ← Test suite
├── docs/                       ← Documentation
├── setup.py                    ← Legacy setup (for compatibility)
├── pyproject.toml              ← Modern build configuration
└── README.md                   ← Project overview
```

## Why src/ Layout?

The `src/` layout is the recommended approach because:

1. **Clean separation**: Keeps source code separate from project configuration
2. **Test isolation**: Tests won't accidentally import from the current directory
3. **Installation clarity**: Forces proper package installation via `pip install -e .`
4. **Industry standard**: Used by major projects (pytest, pip, certifi, etc.)

## Directory Details

### `/src/counterfactual_rl/` - Core Package

This is the main Python package installed via `pip`. It's structured for **environment-agnostic** analysis:

```
src/counterfactual_rl/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   ├── counterfactual.py       ← CounterfactualAnalyzer (works with ANY StateManager)
│   └── metrics.py              ← Metric calculations
├── environments/
│   ├── __init__.py
│   ├── registry.py             ← StateManager registry and factory functions
│   ├── state_manager.py        ← Base StateManager class (interface)
│   ├── frozenlake_manager.py   ← FrozenLake-specific StateManager
│   └── taxi_manager.py         ← Taxi-specific StateManager
├── visualization/
│   ├── __init__.py
│   ├── registry.py             ← Visualizer registry
│   ├── visualization.py        ← Core visualization engines
│   └── plots.py                ← Plotting utilities
└── utils/
    ├── __init__.py
    ├── data_structures.py      ← Custom data structures
    └── helpers.py              ← Helper functions
```

**Key Design Principle**: The `analysis/` and `visualization/` modules don't import environment-specific code directly. They work through the `StateManager` registry.

### `/examples/` - Usage Examples

Demonstration scripts and notebooks showing how to use the package:

```
examples/
├── compare_metrics_analysis.py ← Example: Compare metrics across environments
├── counterfactual_analysis_demo.ipynb ← Jupyter notebook demo
├── basic_usage.ipynb           ← Getting started notebook
└── models/                     ← Pre-trained agent models
    ├── frozenlake_model.pkl
    └── taxi_model.pkl
```

**Note**: All scripts in `/examples/` use the standard import:
```python
from counterfactual_rl.analysis import CounterfactualAnalyzer
from counterfactual_rl.environments.registry import get_environment_manager
```

### `/tests/` - Test Suite

Tests for all core modules:

```
tests/
├── __init__.py
├── test_state_manager.py       ← Tests for StateManager implementations
├── test_analysis.py            ← Tests for analysis module
├── test_visualization.py       ← Tests for visualization
└── test_registry.py            ← Tests for registry system
```

**Running tests**:
```bash
# From project root
pytest
```

### `/docs/` - Documentation

Comprehensive documentation and guides:

```
docs/
├── README.md                   ← Documentation overview
├── algorithm_overview.md       ← How counterfactual analysis works
├── api_reference.md            ← API documentation
├── causal_modeling.md          ← Causal modeling concepts
├── usage_guide.md              ← How to use the package
├── visualization_guide.md      ← Visualization examples
├── REGISTRY_GUIDE.md           ← How to register new environments
├── COUNTERFACTUAL_ANALYZER_REFACTORING.md ← Design decisions
└── README_REGISTRY_SYSTEM.md   ← Registry system details
```

### Configuration Files

**`setup.py`** (Legacy, for compatibility)
- Defines package metadata
- Used by older tools
- Still works but redundant with `pyproject.toml`

**`pyproject.toml`** (Modern, recommended)
- Build system configuration
- Package metadata
- Tool configurations (pytest, black, isort)
- Dependencies

## Import Paths

After installation, use standard imports:

```python
# Import from the package (installed in site-packages or development mode)
from counterfactual_rl.analysis import CounterfactualAnalyzer
from counterfactual_rl.environments.registry import get_environment_manager, register_environment
from counterfactual_rl.visualization import CounterfactualVisualizer
from counterfactual_rl.utils import helpers
```

**NOT**:
```python
# Don't do relative imports like this
from src.counterfactual_rl.analysis import ...  # WRONG

# Don't do this for scripts in examples/
import sys
sys.path.insert(0, '..')  # BAD - works by accident, breaks when installed
```

## Adding New Environments

1. Create a new file: `src/counterfactual_rl/environments/myenv_manager.py`

2. Implement a `StateManager` subclass:
```python
from counterfactual_rl.environments.state_manager import StateManager

class MyEnvStateManager(StateManager):
    def __init__(self, env):
        super().__init__(env)
    
    def get_state_info(self, state):
        # Custom state parsing for your environment
        return {"state": state, "features": {...}}
```

3. Register it in `src/counterfactual_rl/environments/__init__.py`:
```python
from counterfactual_rl.environments.registry import register_environment
from counterfactual_rl.environments.myenv_manager import MyEnvStateManager

register_environment("myenv", MyEnvStateManager)
```

4. Use it anywhere:
```python
from counterfactual_rl.environments.registry import get_environment_manager

manager = get_environment_manager("myenv")
state_manager = manager.create_state_manager(env)
analyzer = CounterfactualAnalyzer(state_manager)
```

## Installation and Development

### For Users
```bash
pip install counterfactual-reasoning
```

### For Development
```bash
git clone <repo>
cd counterfactual-reasoning
pip install -e ".[dev]"
```

The `src/` layout ensures tests work correctly in development mode.

## Migration from Old Structure

Previously, the structure was:
```
counterfactual_frozenlake/
└── counterfactual_rl/  ← Double nesting, hard to reuse
```

Problems:
- Hard to import outside the folder
- Specific to FrozenLake
- Difficult for others to use

New structure solves these:
- Clear package organization
- Environment-agnostic design (via registry)
- Professional Python packaging
- Easy to install and use anywhere

See [MIGRATION.md](docs/MIGRATION.md) for updating old code.

## Best Practices

1. **Keep analysis code environment-agnostic**: Use `StateManager` base class
2. **Add environment-specific code to `environments/`**: Create new `StateManager` subclasses
3. **Register new environments**: Use the registry system
4. **Put examples in `examples/`**: Not mixed in with source code
5. **Test everything**: Add tests when adding features
6. **Document changes**: Update relevant docs when you make changes

## Troubleshooting

**"ModuleNotFoundError: No module named 'counterfactual_rl'"**

→ Did you run `pip install -e .` from the project root?

**"ImportError: cannot import name from counterfactual_rl"**

→ Make sure you're using the new import path, not `from src.counterfactual_rl...`

**Tests fail to import counterfactual_rl**

→ Run tests from project root: `pytest` (not from `src/` directory)

## Summary

| Aspect | Why | Where |
|--------|-----|-------|
| Source code | Core package | `src/counterfactual_rl/` |
| Examples | Usage demos | `examples/` |
| Tests | Quality assurance | `tests/` |
| Docs | Learning material | `docs/` |
| Config | Build & tool setup | `pyproject.toml`, `setup.py` |
| Analysis | Logic | `src/counterfactual_rl/analysis/` |
| Environments | StateManagers | `src/counterfactual_rl/environments/` |
| Visualization | Plots & graphs | `src/counterfactual_rl/visualization/` |

This structure makes the codebase professional, reusable, and maintainable.
