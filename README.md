# Counterfactual Reasoning in Reinforcement Learning

A framework for analyzing **consequential states** in reinforcement learning by asking: *"What if the agent had taken a different action?"*

This tool identifies critical decision points in RL episodes by comparing the distribution of future outcomes under the chosen action versus alternative actions.

## Core Concept

At each timestep, the analyzer:
1. **Saves** the current environment state
2. **Evaluates** multiple alternative actions by rolling out the policy from that state
3. **Compares** the distributions of returns using statistical divergence metrics
4. **Scores** how "consequential" the decision was based on how different the alternatives would have been

A **high consequence score** means: *"There existed an alternative action that would have led to a very different future."*

## Features

- **Multiple Environment Support**
  - FrozenLake (discrete, single-agent)
  - SMAC (StarCraft Multi-Agent Challenge)
  - SMAX (JaxMARL - JAX-accelerated SMAC)

- **Efficient Multi-Agent Analysis**
  - Beam search for top-K joint actions (avoids exponential enumeration)
  - JAX-vectorized rollouts (vmap/jit) for 100x+ speedup

- **Four Divergence Metrics**
  - KL Divergence (asymmetric, unbounded)
  - Jensen-Shannon Divergence (symmetric, bounded)
  - Total Variation Distance (L1 distance)
  - Wasserstein Distance (Earth mover's distance)

- **Flexible Aggregation**
  - `max` - Maximum divergence across alternatives (find the "worst case")
  - `mean` - Average divergence
  - `weighted_mean` - Weighted by action probability (most principled)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd counterfactual-reasoning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# For SMAX support (optional)
pip install jaxmarl
```

## Quick Start

### FrozenLake (Discrete Environment)

```python
from counterfactual_rl.environments import registry
from counterfactual_rl.analysis import CounterfactualAnalyzer
import gymnasium as gym
from stable_baselines3 import PPO

# Setup
env = gym.make("FrozenLake-v1", is_slippery=False)
model = PPO.load("model.zip", env=env)
state_manager = registry.get_state_manager("FrozenLake-v1")

# Analyze
analyzer = CounterfactualAnalyzer(
    model=model,
    env=env,
    state_manager=state_manager,
    horizon=20,      # Rollout steps
    n_rollouts=48    # Samples per action
)
records = analyzer.evaluate_episode()

# Results
for r in records:
    print(f"State {r.state}: KL={r.kl_score:.4f}")
```

### SMAC (Multi-Agent)

```python
from smac.env import StarCraft2Env
from counterfactual_rl.environments.smac import CentralizedSmacWrapper, SmacStateManager
from counterfactual_rl.analysis import MultiDiscreteCounterfactualAnalyzer

# Setup
smac_env = StarCraft2Env(map_name="3m")
wrapped_env = CentralizedSmacWrapper(smac_env)
state_manager = SmacStateManager()

# Analyze with beam search (top-20 actions instead of 9^3=729)
analyzer = MultiDiscreteCounterfactualAnalyzer(
    model=model,
    main_env=wrapped_env,
    state_manager=state_manager,
    top_k=20,
    horizon=20
)
records = analyzer.evaluate_episode()
```

### SMAX/JaxMARL (JAX-Accelerated)

```python
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from counterfactual_rl.analysis.smax_vectorized_counterfactual import (
    SMAXVectorizedCounterfactualAnalyzer
)
from counterfactual_rl.utils.smax_jax_utils import make_jax_random_policy
import jax

# Setup
scenario = map_name_to_scenario("3m")
env = make("HeuristicEnemySMAX", scenario=scenario)
policy_fn = make_jax_random_policy(list(env.agents))

# Vectorized analyzer (JIT-compiled, 100x faster)
analyzer = SMAXVectorizedCounterfactualAnalyzer(
    env=env,
    policy_fn=policy_fn,
    horizon=20,
    n_rollouts=48,
    top_k=20,
    aggregation='weighted_mean'  # 'max', 'mean', or 'weighted_mean'
)

key = jax.random.PRNGKey(42)
records = analyzer.evaluate_episode(key, max_steps=50)

# Save video and plots
analyzer.save_video("episode.gif")
analyzer.save_plots(records, save_dir="./results", n_enemies=3)
```

## Project Structure

```
counterfactual-reasoning/
├── src/counterfactual_rl/
│   ├── analysis/                    # Core algorithms
│   │   ├── counterfactual.py        # Base analyzer (discrete actions)
│   │   ├── multidiscrete_counterfactual.py  # Multi-agent with beam search
│   │   ├── smax_counterfactual.py   # SMAX (non-vectorized)
│   │   ├── smax_vectorized_counterfactual.py  # SMAX (JAX vmap/jit)
│   │   └── metrics.py               # KL, JSD, TV, Wasserstein
│   │
│   ├── environments/                # State management per environment
│   │   ├── base.py                  # Abstract StateManager interface
│   │   ├── frozen_lake.py           # FrozenLake implementation
│   │   ├── smac.py                  # SMAC wrapper + replay strategy
│   │   └── registry.py              # Environment factory
│   │
│   ├── utils/                       # Utilities
│   │   ├── data_structures.py       # ConsequenceRecord dataclass
│   │   ├── smac_data_structures.py  # SmacConsequenceRecord
│   │   ├── smax_data_structures.py  # SMAXConsequenceRecord
│   │   ├── action_selection.py      # Beam search algorithm
│   │   ├── smax_utils.py            # SMAX helpers
│   │   └── smax_jax_utils.py        # JAX-compatible utilities
│   │
│   ├── visualization/               # Plotting
│   │   ├── plots.py                 # Base plotter (heatmaps, histograms)
│   │   ├── smac_plots.py            # SMAC-specific visualizations
│   │   └── smax_plots.py            # SMAX-specific visualizations
│   │
│   ├── policies/                    # Policy adapters
│   │   └── policy_adapters.py       # RandomPolicy, PPO adapters
│   │
│   └── simulations/                 # Example scripts
│       ├── smac_random_policy.py
│       ├── smax_random_policy.py
│       ├── smax_vectorized_random_policy.py
│       ├── run_*.sh                 # SLURM sbatch scripts
│       ├── logs/                    # Job outputs
│       └── runs/                    # Timestamped results
│
├── examples/                        # Jupyter notebooks & demos
├── docs/                            # Additional documentation
├── tests/                           # Unit tests
└── refactorization_cleanup.md       # Planned restructuring notes
```

## Algorithm Details

### Counterfactual Rollout Analysis

```
For each timestep t in episode:
  1. Save state S_t
  2. Get action a_t from policy
  3. For each alternative action a' in top-K:
     - Restore to S_t
     - Execute a'
     - Follow policy for H steps (horizon)
     - Record cumulative discounted return G
     - Repeat N times → distribution D(a')
  4. Compute KL(D(a_t) || D(a')) for each a'
  5. Score = aggregation(KL values)
```

### Beam Search for Multi-Agent Actions

The joint action space for n agents with m actions each is O(m^n) - combinatorially explosive.

**Solution:** Beam search finds top-K most probable joint actions in O(n × K × m).

```python
# 3 agents × 9 actions = 729 combinations
# Beam search with K=20 → only 20 evaluated
actions, probs = beam_search_top_k_joint_actions(
    valid_actions=[[0,1,2,...], [0,1,2,...], [0,1,2,...]],
    k=20,
    return_probs=True  # Also get probabilities for weighted_mean
)
```

### JAX Vectorization (SMAX)

The vectorized analyzer compiles all rollouts into a single XLA kernel:

| Loop | Python (slow) | JAX Primitive | Speedup |
|------|---------------|---------------|---------|
| Actions (outer) | `for action in actions` | `jax.vmap` | Parallel |
| Rollouts (middle) | `for i in range(N)` | `jax.vmap` | Parallel |
| Steps (inner) | `for t in range(H)` | `jax.lax.scan` | Compiled |

Result: **~100-500x speedup** after one-time JIT compilation.

## Output: ConsequenceRecord

Each analyzed timestep produces a record:

```python
SMAXConsequenceRecord(
    timestep=25,
    action=(4, 3, 2),           # Joint action taken
    kl_score=2.34,              # Consequence score (max KL)
    kl_divergences={            # KL vs each alternative
        (0, 3, 2): 2.34,
        (4, 0, 2): 0.12,
        ...
    },
    return_distributions={      # Raw return samples
        (4, 3, 2): array([1.2, 0.8, ...]),  # N samples
        (0, 3, 2): array([0.1, 0.0, ...]),
        ...
    },
    jsd_score=0.42,
    tv_score=0.78,
    wasserstein_score=0.31
)
```

## Metrics Explained

| Metric | Range | Properties | Best For |
|--------|-------|------------|----------|
| **KL Divergence** | [0, ∞) | Asymmetric, sensitive to tail differences | Detecting any difference |
| **Jensen-Shannon** | [0, ln(2)] | Symmetric, bounded | Balanced comparison |
| **Total Variation** | [0, 1] | Symmetric, bounded, L1-based | Binary "same or different" |
| **Wasserstein** | [0, ∞) | Symmetric, geometric interpretation | Magnitude of difference |

## Aggregation Methods

How to combine divergences across alternative actions into a single score:

- **`max`** (default for critical decisions): "What's the worst-case alternative?"
- **`mean`**: "On average, how different are alternatives?"
- **`weighted_mean`**: "Weighted by how likely I was to take each alternative" — most principled for policy analysis

## Running on SLURM (HPC)

```bash
cd src/counterfactual_rl/simulations
sbatch run_SMAX_vectorized_analysis.sh
```

Results saved to `runs/smax_vectorized_run_<timestamp>/`:
- `episode_replay.gif` - Animated episode replay
- `analysis_comprehensive.png` - Multi-panel analysis figure
- `consequence_over_time.png` - Metrics timeline
- `*.log` - Detailed logs

## Extending to New Environments

1. **Create a StateManager** (in `environments/`):
   ```python
   class MyEnvStateManager(StateManager):
       @staticmethod
       def clone_state(env) -> Dict:
           return {"position": env.agent_pos, "rng": env.np_random.get_state()}

       def restore_state(env, state_dict):
           env.agent_pos = state_dict["position"]
           env.np_random.set_state(state_dict["rng"])
   ```

2. **Register it**:
   ```python
   from counterfactual_rl.environments import registry
   registry.register("MyEnv-v0", MyEnvStateManager, MyEnvConfig)
   ```

3. **Use existing analyzers** - no changes needed!

## Documentation

- `docs/COUNTERFACTUAL_COMPONENTS.md` - Detailed method breakdown
- `docs/MULTIDISCRETE_COUNTERFACTUAL.md` - Beam search explanation
- `refactorization_cleanup.md` - Planned code restructuring

## Citation

If you use this code in your research, please cite:

```bibtex
@software{counterfactual_rl,
  title = {Counterfactual Reasoning in Reinforcement Learning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

## License

MIT
