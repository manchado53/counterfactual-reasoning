# CCE Hyperparameter Tuning Guide

Parameters to consider carefully when applying CCE to a new environment.
These are the knobs most likely to need environment-specific adjustment.

---

## Core CCE Parameters

### `cf_horizon` — Counterfactual rollout horizon
How many steps each counterfactual rollout runs from a scored transition.

**Why it matters:** If the horizon is too short, rollouts rarely reach a terminal state and the divergence signal is weak. If too long, rollouts are expensive and may diverge for reasons unrelated to the scored transition.

**Rule of thumb:** Set to the typical number of steps between a consequential decision and its outcome. For sparse-reward environments, longer is better. For dense-reward environments, shorter is fine.

| Environment | Value | Rationale |
|---|---|---|
| SMAX 3m / 8m | 30 | Multi-agent battles resolve in 30-60 steps; 30 captures most of the consequence window |
| FrozenLake 8×8 | 20 | Optimal paths are ~14-20 steps; 20 covers typical path length with slipping |
| Gardner Chess | 10 | Each step = full white+black pair; games are short; opponent inference expensive |

**Max episode length reference:**
- FrozenLake 8×8: 200 steps (Gymnasium default)
- SMAX 3m: ~150 steps typical
- Chess: variable, typically 20-60 moves

---

### `cf_n_rollouts` — Number of rollouts per scored transition
How many independent rollouts to sample per (state, action) pair when estimating the counterfactual distribution.

**Why it matters:** More rollouts = lower variance in the divergence estimate, but linear compute cost. In stochastic environments (FrozenLake slippery, SMAX unit noise) you need more rollouts to get a stable distribution estimate.

| Environment | Value | Rationale |
|---|---|---|
| SMAX 3m / 8m | 30 | High stochasticity from multi-agent interactions |
| FrozenLake 8×8 | 20 | Moderate stochasticity from slipping |
| Gardner Chess | 16 | Opponent policy is deterministic (AlphaZero baseline); lower variance |

---

### `score_interval` — How often to run the CCE scoring pass
Score every N Q-network updates. Lower = more frequent scoring = better signal, more compute.

**Why it matters:** CCE scores transitions in the replay buffer. If you score too rarely, many transitions have stale scores. If you score every update, the overhead dominates training.

**Rule of thumb:** Score often enough that most transitions in the active training batch have been scored at least once per ~1,000 episodes.

| Environment | Value | Rationale |
|---|---|---|
| SMAX 3m / 8m | 200 | ~hundreds of transitions per episode; scoring 200 updates ≈ every few episodes |
| FrozenLake 8×8 | 100 | Short episodes; need to score frequently to keep buffer fresh |
| Gardner Chess | 1000 | 65k transitions per chunk; buffer fills slowly; less frequent scoring sufficient |

---

### `n_score_sample` — Transitions scored per scoring pass
How many transitions to sample from the buffer and score each time CCE runs.

**Why it matters:** Larger = more of the buffer covered per pass, but proportionally more compute per scoring call.

| Environment | Value | Rationale |
|---|---|---|
| SMAX 3m / 8m | 256 | Large buffer; need to cover it efficiently |
| FrozenLake 8×8 | 128 | Smaller effective buffer; 128 sufficient |
| Gardner Chess | 128 | Buffer fills slowly; 128 per pass is adequate |

---

### `mu` — Priority mixing weight
Controls the blend between TD-error priority (standard PER) and CCE consequence score.
- `mu=0.0` → pure TD-error (equivalent to DQN+PER)
- `mu=1.0` → pure CCE consequence score (DQN+CCE-only)
- `mu=0.25` → 75% TD, 25% CCE (selected by sweep as best for SMAX)

**Selected value:** `mu=0.25` via two-phase hyperparameter sweep on SMAX 3m.
Apply this value to all environments unless environment-specific sweep suggests otherwise.

---

### `consequence_metric` — Divergence metric for scoring
How to measure the divergence between actual and counterfactual return distributions.

**Selected value:** `total_variation` — selected by Phase 1 sweep on SMAX 3m (69.2% win rate vs wasserstein at 60.0%). Apply to all environments.

Options: `total_variation`, `kl_divergence`, `jensen_shannon`, `wasserstein`

---

## Environment-Specific Considerations

### Sparse vs dense reward
CCE signal is strongest in sparse-reward environments (FrozenLake, Chess) where most transitions have zero reward and the few consequential ones are hard to find. In dense-reward environments (SMAX shaped reward) the TD-error signal is already informative, so CCE adds less marginal value.

### Episode length
Longer episodes → use longer `cf_horizon`. Shorter episodes → can use shorter horizon without losing coverage.

### Stochasticity
More stochastic environments need more `cf_n_rollouts` to get stable distribution estimates.

### Transition throughput
High-throughput environments (Chess: 65k transitions/chunk) need higher `score_interval` to avoid scoring dominating compute. Low-throughput environments (FrozenLake: ~100 transitions/episode) need lower `score_interval` to keep scores fresh.

---

## Current Values by Environment

| Parameter | SMAX 3m/8m | FrozenLake 8×8 | Gardner Chess |
|---|---|---|---|
| `cf_horizon` | 30 | 20 | 10 |
| `cf_n_rollouts` | 30 | 20 | 16 |
| `score_interval` | 200 | 100 | 1000 |
| `n_score_sample` | 256 | 128 | 128 |
| `mu` | 0.25 | 0.25 | 0.25 |
| `consequence_metric` | total_variation | total_variation | total_variation |
| `eval_episodes` | 100 | 100 | 50 |
