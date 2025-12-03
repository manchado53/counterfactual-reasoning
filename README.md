# Counterfactual Reasoning in Reinforcement Learning

A framework for analyzing consequential states in RL using counterfactual reasoning.

## Features

- **Environment-agnostic**: Works with FrozenLake, SMAC, and any Gym environment
- **Counterfactual analysis**: Identifies which states had the most impact on outcomes
- **Multiple metrics**: KL divergence, Jensen-Shannon, Total Variation, Wasserstein
- **SMAC integration**: Replay-based state restoration for StarCraft II

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd counterfactual-reasoning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Quick Start

### FrozenLake Example
```python
from counterfactual_rl.environments import registry
from counterfactual_rl.analysis.counterfactual import CounterfactualAnalyzer

# Get environment components
state_manager = registry.get_state_manager("FrozenLake-v1")

# Run analysis
analyzer = CounterfactualAnalyzer(model, env, state_manager)
records = analyzer.evaluate_episode()
```

### SMAC Example
See `examples/smac_counterfactual_demo.ipynb`

## Training an Agent

```bash
python train_ppo_smac.py
```

## Project Structure

```
counterfactual-reasoning/
├── src/counterfactual_rl/
│   ├── analysis/          # Counterfactual analysis
│   ├── environments/      # Environment wrappers
│   └── utils/            # Utilities
├── examples/             # Demo notebooks
├── tests/               # Unit tests
└── train_ppo_smac.py    # Training script
```

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```

## License

MIT
