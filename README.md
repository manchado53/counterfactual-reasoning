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

- **Consequence-weighted DQN Training (Algorithm 2)**
  - Full training pipeline integrating counterfactual consequence scoring into DQN
  - Prioritized Experience Replay weighted by both TD-error and consequence score
  - Additive (Eq 4) and multiplicative (Eq 5) priority mixing modes
  - Circular replay buffer for O(1) add/eviction (no per-step memory shifts)
  - SLURM experiment sweep infrastructure for hyperparameter search

- **Efficient Multi-Agent Analysis**
  - Beam search for top-K joint actions (avoids exponential enumeration)
  - JAX triple-vmap rollouts (transitions × actions × rollouts) compiled via JIT

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

## Training Consequence-weighted DQN

### Running a single training job locally

```bash
conda activate counterfactual
cd src/counterfactual_rl/training/smax/shared

CONFIG_OVERRIDES='{"scenario":"3m","n_episodes":1000,"algorithm":"consequence-dqn"}' \
    python -m counterfactual_rl.training.smax.shared.train
```

Results are written to `runs/<job_id>/`:
- `timing.jsonl` — per-component timing records
- `metrics.log` — win rate, return, episode length per eval interval
- `training_curves.png` — return and episode length over training
- `timing_breakdown.png` / `timing_timeseries.png` — profiling plots
- `last.pkl` / `best.pkl` — saved model weights

### Running experiment sweeps on SLURM

```bash
cd src/counterfactual_rl/training/smax/shared

# Preview jobs without submitting
python run_experiments.py metric_sweep --dry-run

# Submit all jobs
python run_experiments.py metric_sweep
python run_experiments.py algorithm_comparison
python run_experiments.py mu_sweep
```

Available experiments: `smoke_test`, `metric_sweep`, `algorithm_comparison`, `mu_sweep`.

### Key configuration parameters

| Parameter | Default | Description |
|---|---|---|
| `scenario` | `3s5z` | SMAX map: `3m`, `2s3z`, `5m_vs_6m`, `3s5z`, etc. |
| `algorithm` | `consequence-dqn` | `dqn-uniform`, `dqn`, or `consequence-dqn` |
| `mu` | `0.5` | Priority blend: 0 = pure TD, 1 = pure consequence |
| `priority_mixing` | `additive` | `additive` (Eq 4) or `multiplicative` (Eq 5) |
| `consequence_metric` | `wasserstein` | `kl_divergence`, `jensen_shannon`, `total_variation`, `wasserstein` |
| `score_interval` | `200` | Q-updates between consequence scoring passes |
| `cf_horizon` | `30` | Counterfactual rollout horizon |
| `cf_n_rollouts` | `30` | Rollouts per action per transition |
| `cf_top_k` | `10` | Top-K actions from beam search |
| `M` | `100000` | Replay buffer capacity |

### Running on SLURM (HPC)

```bash
cd src/counterfactual_rl/training/smax/shared
sbatch train_smax_dqn.sh
```

Legacy counterfactual analysis scripts:

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

## Known Bugs and Defects

See `bugs.md` for full detail. Summary of open issues:

| # | File | Severity | Description |
|---|---|---|---|
| 1 | `consequence_diagnostics.py:118` | Low | `buffer_scored_frac` always reports ~100% — uses wrong sentinel (`max_priority`) instead of initial consequence score (`0.0`) |
| 2 | `consequence_dqn.py:76` | Low | `diagnostics_enabled` fallback is `True` but `DEFAULT_CONFIG` default is `False` — partial configs behave unexpectedly |
| 4 | `config.py:80` | Low | `3s5z` preset uses `hidden_dim=516` — likely typo for `512` (all other presets use powers of 2) |
| 6 | `experiments.py:116` | Cosmetic | Run count comment (`57 runs`) is stale after `MU_SWEEP` was narrowed to `3m` only |
| 7 | `consequence_buffers.py:152` | Medium | IS weights not normalized by batch max — rare high-priority transitions can produce very large weights, potentially destabilizing updates |
| 8 | `consequence_dqn.py:208` | Low | Duplicate action padding for under-full action sets wastes JIT compute (rolled-out returns discarded silently) |

**Fixed:**
- Bug 3: `ConsequenceDQN.learn()` now seeds NumPy RNG for reproducibility
- Bug 5: `ConsequenceReplayBuffer.add()` replaced `list.pop(0)` (O(N)) with circular buffer (O(1)) — eliminated 43% of total training time in profiling

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
