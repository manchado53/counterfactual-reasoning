# CounterfactualAnalyzer Components

This document tracks the main components of `counterfactual.py` for ongoing development.

---

## Class: `CounterfactualAnalyzer`

Performs counterfactual rollout analysis to identify consequential state-action pairs.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `PPO` | required | Trained Stable Baselines3 PPO model |
| `env` | `gym.Env` | required | Gymnasium environment |
| `state_manager` | `StateManager` | required | Environment-specific state manager for cloning/restoration |
| `horizon` | `int` | 20 | Number of steps to roll out policy |
| `n_rollouts` | `int` | 48 | Number of rollouts per action for distribution estimation |
| `gamma` | `float` | 0.99 | Discount factor for returns |
| `deterministic` | `bool` | True | Use deterministic (argmax) vs stochastic policy |

### Instance Attributes

- `action_space_size`: Number of discrete actions (from `env.action_space.n`)

---

## Core Methods

### 1. `perform_counterfactual_rollouts(state_dict)`
**Purpose**: Perform rollouts for all possible actions from a saved state.

**Current Implementation**:
```python
for action in range(self.action_space_size):  # ⚠️ Exhaustive enumeration
    for _ in range(self.n_rollouts):
        1. Restore environment to state_dict
        2. Execute counterfactual action
        3. Roll out policy for (horizon - 1) steps
        4. Accumulate discounted return
```

**Returns**: `Dict[int, np.ndarray]` - Maps action → array of returns (shape: `n_rollouts`)

> [!IMPORTANT]
> **SCALING ISSUE**: This exhaustively iterates over all actions, which doesn't scale for large/continuous action spaces (e.g., SMAC's `MultiDiscrete`).

---

### 2. `compute_consequence_score(action, return_distributions)`
**Purpose**: Compute the consequence score using KL divergence.

**Logic**:
1. For each alternative action, compute KL divergence between chosen action's returns and alternative's returns
2. Return `max(kl_divergences)` as the consequence score

**Returns**: `(float, Dict[int, float])` - (max_kl_score, {action: kl_value})

---

### 3. `compute_all_metrics(action, return_distributions)`
**Purpose**: Compute all distributional metrics (KL, JSD, TV, Wasserstein).

**Returns**:
```python
{
    'kl': {'score': float, 'divergences': Dict[int, float]},
    'jsd': {'score': float, 'divergences': Dict[int, float]},
    'tv': {'score': float, 'distances': Dict[int, float]},
    'wasserstein': {'score': float, 'distances': Dict[int, float]}
}
```

---

### 4. `evaluate_episode(max_steps, verbose, compute_all_metrics)`
**Purpose**: Evaluate consequential states for a single episode.

**Flow**:
```
reset env
while not done and step < max_steps:
    1. Save current state (clone_state)
    2. Get action from policy (model.predict)
    3. Perform counterfactual rollouts for ALL actions
    4. Compute consequence metrics
    5. Create ConsequenceRecord
    6. Restore state, execute chosen action, advance
```

**Returns**: `List[ConsequenceRecord]`

---

### 5. `evaluate_multiple_episodes(n_episodes, verbose, compute_all_metrics)`
**Purpose**: Run `evaluate_episode` across multiple episodes.

**Returns**: `List[ConsequenceRecord]` - All records from all episodes

---

## Dependencies

### Imports
- `StateManager` from `counterfactual_rl.environments.base`
- `ConsequenceRecord` from `counterfactual_rl.utils.data_structures`
- Metrics from `counterfactual_rl.analysis.metrics`:
  - `compute_kl_divergence_kde`
  - `compute_jensen_shannon_divergence`
  - `compute_total_variation`
  - `compute_wasserstein_distance`

---

## Extension Points

| What to extend | How |
|----------------|-----|
| New environment | Create new `StateManager` subclass, register with `registry` |
| New metrics | Add to `counterfactual_rl.analysis.metrics`, update `compute_all_metrics()` |
| Action selection | **Modify `perform_counterfactual_rollouts()`** ← This is what we're changing |

---

## Current Action Selection Strategy

```
Strategy: EXHAUSTIVE ENUMERATION
─────────────────────────────────
for action in range(action_space_size):  # ALL actions
    perform n_rollouts for this action
```

**Problem**: For environments like SMAC with `MultiDiscrete(6, 6, 6, ...)` action spaces, this is combinatorially explosive.

**Proposed Change**: Select only top-K most probable actions according to PPO policy.
