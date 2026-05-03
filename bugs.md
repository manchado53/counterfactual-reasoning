# Bug Report

---

## Bug 1 — Wrong "unscored" sentinel in diagnostics (logic error)

**File:** `src/counterfactual_rl/training/smax/shared/consequence_diagnostics.py:118`

```python
default_cscore = buffer.max_priority  # initial value assigned on add
scored_mask = buf_cscores != default_cscore
```

The comment claims `max_priority` is the initial consequence score, but `ConsequenceReplayBuffer.add()` initializes it to `mean(existing_scores)` or `0.0` — never `max_priority`. As a result, `scored_mask` is almost always all-`True` (no score equals `1.0` by coincidence), making `buffer_scored_frac` report ~100% scored from the very start, even before any scoring pass has run.

**Root cause:** The sentinel used here (`buffer.max_priority`) is the initial TD magnitude default, not the initial consequence score default.

---

## Bug 2 — `diagnostics_enabled` fallback inconsistent with config default

**File:** `src/counterfactual_rl/training/smax/dqn_jax/consequence_dqn.py:76`

```python
self.diagnostics_enabled = self.config.get('diagnostics_enabled', True)
```

The fallback is `True`, but `DEFAULT_CONFIG` sets it to `False`. In normal usage `self.config` is populated from `DEFAULT_CONFIG`, so the key is always present and the fallback never fires. But if someone constructs `ConsequenceDQN` with a partial config dict that omits `diagnostics_enabled`, they'd unexpectedly get diagnostics enabled (expensive), contradicting the documented default.

---

## Bug 3 — `ConsequenceDQN.learn()` never seeds NumPy RNG (reproducibility break) ✓ FIXED

**File:** `src/counterfactual_rl/training/smax/dqn_jax/consequence_dqn.py` (learn() method)
**Reference:** `src/counterfactual_rl/training/smax/dqn_jax/dqn.py:256`

`DQN.learn()` seeds NumPy's global RNG:
```python
np.random.seed(self.config.get('seed', 0))
```

`ConsequenceDQN.learn()` fully overrides `DQN.learn()` and never calls `np.random.seed()`. All NumPy-based randomness in `ConsequenceDQN` — epsilon-greedy action selection (`np.random.uniform()`, `np.random.choice()`), buffer sampling (`np.random.choice()`), and uniform scoring samples — is therefore not seeded, breaking reproducibility even when `seed` is set.

---

## Bug 4 — Typo in `3s5z` hidden_dim preset (likely off-by-4)

**File:** `src/counterfactual_rl/training/smax/shared/config.py:80`

```python
'3s5z': {'hidden_dim': 516, ...}
```

Every other scenario uses a power-of-2 hidden dimension (128, 192, 256, 512). `516` is almost certainly a typo for `512`. Powers of 2 are standard for GPU efficiency; `516` will still run but wastes alignment and diverges from the pattern of every other preset.

---

## Bug 5 — O(N) FIFO eviction on full replay buffer (performance) ✓ FIXED

**File:** `src/counterfactual_rl/training/smax/dqn_jax/consequence_buffers.py:88-93`

```python
if len(self.buffer) > self.capacity:
    self.buffer.pop(0)
    self.consequence_scores.pop(0)
    self.td_magnitudes.pop(0)
    self.jax_states.pop(0)
    self.jax_obs.pop(0)
```

`list.pop(0)` is O(N) — it shifts every element left. With `capacity=100000`, once the buffer is full, every single `add()` call triggers 5 O(N) shifts (500k element moves per step). This becomes a significant fraction of total training time, especially on scenarios with long episodes. A `collections.deque(maxlen=capacity)` would give O(1) eviction.

---

## Bug 6 — `mu_sweep` comment says 12 runs but only produces 12 (count comment wrong)

**File:** `src/counterfactual_rl/training/smax/shared/experiments.py:116`

```python
# Total: 36 + 12 + 9 = 57 runs (+ 2 smoke test + 6 mixing comparison)
```

`MU_SWEEP` sweeps `mu=[0.25, 0.5, 0.75, 1.0]` × `scenario=['3m']` × `seed=[0,1,2]` = **12 runs**, but the original comment says "36 + 12 + 9 = 57". The 36-run `METRIC_SWEEP` was not reduced when `MU_SWEEP` was narrowed to only `3m`, so the tally is stale. Minor, but misleading when planning compute budget.

---

## Bug 7 — IS weights not normalized (potential training instability)

**File:** `src/counterfactual_rl/training/smax/dqn_jax/consequence_buffers.py:152`

```python
weights = 1.0 / (probs[indices] * N)
```

Standard PER normalizes IS weights by dividing by the max weight in the batch so the largest weight is 1.0, preventing gradient explosion from rare high-priority transitions. Here the raw unnormalized weight `1/(p*N)` is passed directly to the loss. For a very low-probability transition, this weight could be very large (e.g., if `p ≈ 1e-5` and `N=100000`, weight = 1000), destabilizing updates. The `beta` parameter stored on the buffer is used only for priority exponentiation, not for IS weight annealing as in the original PER paper.

---

## Bug 9 — Key generation loop makes B separate JAX calls in scoring stack (performance) ✓ FIXED

**File:** `src/counterfactual_rl/training/smax/dqn_jax/consequence_dqn.py:238-245`

```python
all_keys = []
for b_key in batch_keys:                          # Python loop over B=256 transitions
    action_keys = jax.random.split(b_key, K)      # JAX call per iteration
    rollout_keys = jax.vmap(
        lambda k: jax.random.split(k, N)
    )(action_keys)
    all_keys.append(rollout_keys)
keys_array = jnp.stack(all_keys, axis=0)          # (B, K, N, 2)
```

Profiling on a 25k-episode run shows `update.scoring.stack` takes **752s (6.1% of total runtime)**. The Python loop over B dispatches B separate JAX kernel calls rather than one. This can be replaced with a single vectorized call:

```python
keys_flat = jax.random.split(subkey, B * K * N)
keys_array = keys_flat.reshape(B, K, N, 2)
```

---

## Bug 8 — Duplicate action padding can skew return distribution

**File:** `src/counterfactual_rl/training/smax/dqn_jax/consequence_dqn.py:208-209`

```python
while len(actions_to_eval) < self.cf_top_k:
    actions_to_eval.append(actual_action)
```

When fewer than `cf_top_k` valid actions exist, the list is padded by repeating `actual_action`. In `_compute_priorities`, the return distributions dict deduplicates on action tuple:

```python
if action_tuple not in return_distributions:
    return_distributions[action_tuple] = returns_np[i, j]
```

So the duplicate slots are silently ignored in the metric computation — that part is safe. But the rolled-out returns for those padded slots (indices `j` beyond the real actions) are computed in the JIT call and wasted. More importantly, `returns_np[i, j]` for duplicate-action slots uses a *different* RNG key than the first occurrence of that action (different `j`), so if the code were ever changed to not deduplicate, it would silently use mismatched return samples for the same action.
