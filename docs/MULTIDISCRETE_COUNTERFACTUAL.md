# MultiDiscreteCounterfactualAnalyzer

Counterfactual analysis for MultiDiscrete action spaces (e.g., SMAC multi-agent environments).

---

## Overview

For environments with **MultiDiscrete** action spaces (like SMAC), the joint action space is exponentially large:
- 3 agents × 9 actions = **729** joint actions
- 8 agents × 14 actions = **1.5 billion** joint actions

This class uses **beam search** to efficiently select only the top-K most probable joint actions for counterfactual rollouts.

---

## Usage

```python
from counterfactual_rl.analysis import MultiDiscreteCounterfactualAnalyzer

analyzer = MultiDiscreteCounterfactualAnalyzer(
    model=your_ppo_model,
    env=smac_wrapper,
    state_manager=smac_state_manager,
    get_logits_fn=lambda obs: your_ppo.get_action_prob(obs).view(n_agents, n_actions),
    n_agents=3,
    n_actions=9,
    top_k=20
)

# Evaluate a single episode
records = analyzer.evaluate_episode(max_steps=100, verbose=True)

# Or evaluate multiple episodes
records = analyzer.evaluate_multiple_episodes(n_episodes=20)
```

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | Policy model with `.predict(obs, deterministic)` method returning `(actions, _)` |
| `env` | gym.Env | required | Environment with MultiDiscrete action space |
| `state_manager` | StateManager | required | For cloning/restoring environment state |
| `get_logits_fn` | Callable | required | Function: obs → logits tensor `(n_agents, n_actions)` |
| `n_agents` | int | required | Number of agents |
| `n_actions` | int | required | Number of actions per agent |
| `horizon` | int | 20 | Rollout horizon |
| `n_rollouts` | int | 48 | Rollouts per action for distribution estimation |
| `gamma` | float | 0.99 | Discount factor |
| `deterministic` | bool | True | Use deterministic policy during rollouts |
| `top_k` | int | 20 | Number of top joint actions to evaluate |


---

## Key Component: `get_logits_fn`

This is how you plug in your PPO's action probabilities. It should:
- Take an observation (numpy array)
- Return logits tensor of shape `(n_agents, n_actions)`

### Example for Custom PPO

```python
def get_logits(obs):
    with torch.no_grad():
        logits = my_ppo.actor(torch.tensor(obs))  # Shape: (n_agents * n_actions,)
        logits = logits.view(n_agents, n_actions)  # Reshape
    return logits

analyzer = MultiDiscreteCounterfactualAnalyzer(
    ...
    get_logits_fn=get_logits,
    ...
)
```

---

## Methods

### `perform_counterfactual_rollouts(state_dict, obs)`
Perform rollouts for top-K most probable joint actions.

**Returns**: `Dict[Tuple[int, ...], np.ndarray]` - Maps joint action → array of returns

### `compute_consequence_score(action, return_distributions)`
Compute the consequence score (max KL divergence vs alternatives).

**Returns**: `(float, Dict)` - Score and divergences dict

### `evaluate_episode(max_steps, verbose)`
Evaluate one episode, returning `List[ConsequenceRecord]`.

### `evaluate_multiple_episodes(n_episodes, verbose)`
Evaluate multiple episodes, returning combined records.

---

## Beam Search Algorithm

Located in: `counterfactual_rl/utils/action_selection.py`

```
Complexity: O(n_agents × K × n_actions)
vs Exhaustive: O(n_actions^n_agents)
```

### How it works:
1. Get top-K actions for agent 0
2. For each subsequent agent:
   - Extend each beam with all valid actions
   - Compute cumulative log-probability
   - Prune to top-K beams
3. Return top-K joint actions

---

## Action Masking

Automatically detects SMAC-style action masking:

```python
# Looks for these methods on env:
env.get_avail_actions()          # Returns (n_agents, n_actions) mask
env.env.get_avail_actions()      # If wrapped
```

Invalid actions are masked out during beam search.

---

## Files

| File | Description |
|------|-------------|
| `analysis/multidiscrete_counterfactual.py` | Main analyzer class |
| `utils/action_selection.py` | Beam search function |
