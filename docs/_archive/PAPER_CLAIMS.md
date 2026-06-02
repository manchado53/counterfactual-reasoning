# Paper Claims & Strategy

## Two Core Claims

### Claim 1 — Counterfactual scoring identifies consequential moments

The algorithm assigns higher priority to transitions where the action choice actually
mattered — measured by rolling out K counterfactual alternatives and computing the
divergence between the actual action's return distribution and the alternatives.

**What we need**: Ground-truth labels for "consequential moments" to validate the
scores correlate with real importance. Options:

- **Chess oracle labels** (most convincing): The pgx baseline model outputs a position
  evaluation (value head). For any stored state, query it to get an objective importance
  score. High-consequence positions by the oracle should correlate with high consequence
  scores from our algorithm. Feasible in a few hours.
- **Hindsight SMAX labels**: After episodes complete, label transitions from the final
  20% of winning episodes as "high consequence" (the decisive phase). Check whether
  the scoring mechanism promotes those. Rough but interpretable.
- **Synthetic ground truth**: A simple 2-option environment where the pivotal timestep
  is known. Clean but may feel like a toy result.

**Open question**: Which label source to use — needs advisor input.

---

### Claim 2 — It leads to more sample-efficient early training

Consequence-DQN does not consistently beat baselines at the *end* of training, but it
reaches competent play faster. This is a legitimate and valued result — sample
efficiency matters in practice.

**Current evidence**:
- Multiplicative mixing: 48% win rate at 10k episodes vs 33% for baseline (mixing
  comparison experiment)
- Additive mixing: best final mean (71.7%) but high variance across seeds

**What we need to support it**:
- Clean learning curves across all 4 algorithms (DQN-Uniform, DQN-PER, additive,
  multiplicative), early phase zoomed in (episodes 0–10k)
- A scalar metric: episodes to first reach X% win rate, or AUC over first N episodes
- Enough seeds for statistical comparison (10-seed experiment exists in codebase)

---

## Current Experimental State

| Experiment | Status | Notes |
|---|---|---|
| SMAX 3m, 3 seeds | Done | Some seeds ran on CPU (timing invalid, performance OK) |
| SMAX 3m, 10 seeds | Not run | `FULL_ALGORITHM_COMPARISON_10SEEDS` defined in experiments.py |
| SMAX 8m, 1 seed | Unclear | Jobs 241033–241036 from ~4 weeks ago |
| Chess full training | Cancelled | Job 242922 cancelled at 82%; no win/loss numbers |
| Scatter diagnostic | Running | Jobs 252865–252866, ~1h in; gives visualization for Claim 1 |

---

## What's Most Urgent (paper due next week)

1. **Decide on Claim 1 label source** — chess oracle is probably best
2. **Run 10-seed SMAX experiment** for Claim 2 statistical backing
3. **Get chess full training run** to completion
4. **Extract early-training learning curves** from existing data
