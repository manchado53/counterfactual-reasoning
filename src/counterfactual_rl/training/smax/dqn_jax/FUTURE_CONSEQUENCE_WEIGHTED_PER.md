# Future: Consequence-Weighted Prioritized Experience Replay

## Overview

After the PureJaxRL training loop is working with uniform sampling, integrate
the counterfactual consequence estimation (Algorithm 1) into the replay
priority (Algorithm 2).

## Priority Function

```
p(j) = mu * p_C(j) + (1 - mu) * p_delta(j)
```

- `p_C(j)`: Consequence score — how much the return distribution changes if
  a different action were taken at transition j. Computed by
  `SMAXVectorizedCounterfactualAnalyzer`.
- `p_delta(j)`: TD error priority — how poorly the Q-network models this
  transition. Already computed during gradient steps.
- `mu`: Mixing coefficient (hyperparameter, 0-1).

## Architecture

```
Python outer loop:
  for chunk in range(num_chunks):

      # FAST: JIT-compiled training with weighted sampling (~100 update steps)
      runner_state, metrics = jit_scan(runner_state)

      # PERIODIC: Score buffer with counterfactual consequences
      if chunk % score_interval == 0:
          policy_fn = agent.make_policy_fn()
          indices, scores = score_buffer_sample(buffer_state, policy_fn, analyzer)
          runner_state = update_consequence_scores(runner_state, indices, scores, mu)
```

## Buffer Changes

Add three arrays to `ReplayBufferState`:

```python
priorities: jnp.ndarray           # (capacity,) combined p(j)
td_errors: jnp.ndarray            # (capacity,) latest |delta_j|
consequence_scores: jnp.ndarray   # (capacity,) latest m_C(j)
```

Replace `jax.random.randint` (uniform) with `jax.random.choice` (weighted):

```python
def sample(state, key, batch_size):
    probs = state.priorities[:state.size]
    probs = probs / probs.sum()
    indices = jax.random.choice(key, state.size, shape=(batch_size,), p=probs, replace=False)
    # Importance sampling weights for bias correction
    weights = 1.0 / (state.size * probs[indices])
    weights = weights / weights.max()
    return jax.tree.map(lambda buf: buf[indices], state.data), indices, weights
```

## Scoring Pass (Algorithm 1)

Every `score_interval` chunks, sample ~1000 transitions from the buffer and
run the vectorized counterfactual analyzer on their states:

```python
def score_buffer_sample(buffer_state, policy_fn, env, analyzer, n_sample=1000):
    sample = JaxReplayBuffer.sample(buffer_state, key, n_sample)
    # For each state, run counterfactual rollouts across all actions
    # Uses SMAXVectorizedCounterfactualAnalyzer (vmap actions x vmap rollouts x scan steps)
    scores = analyzer.batch_evaluate(sample['state'], policy_fn)
    return sample_indices, scores
```

Write scores back to `buffer_state.consequence_scores[indices]`, then
recompute priorities as `mu * consequence + (1-mu) * td_error`.

## TD Error Updates

Inside the JIT-compiled gradient step, after computing `td_errors`:

```python
buffer_state = buffer_state.replace(
    td_errors=buffer_state.td_errors.at[sampled_indices].set(jnp.abs(td_errors)),
    priorities=mu * buffer_state.consequence_scores[sampled_indices]
              + (1 - mu) * (jnp.abs(td_errors) + eps) ** beta,
)
```

## Key Hyperparameters

- `mu`: Consequence vs TD-error weight (start with 0.5)
- `score_interval`: Chunks between consequence scoring passes (start with 10)
- `n_score_sample`: Transitions to score per pass (start with 1000)
- `beta`: Priority exponent (0.25, same as current PER)
- `eps`: Priority smoothing (0.01, same as current PER)

## Why This Works

1. Training loop stays fast — weighted sampling is O(1) with `jax.random.choice`
2. Consequence scoring is amortized — run every K chunks, not every step
3. Stale scores are acceptable — they're approximations of importance
4. Reuses existing `SMAXVectorizedCounterfactualAnalyzer` unchanged
5. H100 benefits from both fast training AND fast counterfactual rollouts
