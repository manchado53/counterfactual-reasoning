# Claim 1 — Metrics Cookbook

> **How we measure "CCE finds the consequential moments."** Definitions, formulas, and
> how-to-compute only. **No results or numbers here** — current results live in
> `/lab-notebook.md` (status) and `/paper/paper.tex` (the paper).

## What Claim 1 tests
CCE scores — computed only from the agent's own rollouts — should rank states the same way
an independent ground-truth oracle does. Agreement (ρ > 0) shows rollout-based CCE recovers
real importance without needing a value function.

## Environment requirement
Needs a **trustworthy oracle**: an independent ground-truth importance per state.
- FrozenLake has it exactly (value iteration on the known MDP).
- Measure on a **deterministic** env — stochastic dynamics blur each action's outcome
  distribution, so the CCE score then reflects noise instead of the decision.
- Most envs lack a clean oracle — that's the bottleneck for adding Claim-1 scenarios.

## The CCE score (the thing under test)
For each state `s`: roll out the current greedy policy under each action `a`; estimate the
return distribution per action; score = **total-variation** divergence among those
distributions (aggregation = `weighted_mean`).
Code: `analysis/claim1/frozen_lake/score_states.py`, `analysis/metrics.py:compute_consequence_metric`.

## The oracle (ground truth — FrozenLake)
Value iteration on `env.P` → `Q*(s,a)`:
```
Q*(s,a)   = Σ_s'  P(s'|s,a) [ R(s,a,s') + γ V*(s') ]
a*        = argmax_a Q*(s,a)
oracle(s) = mean_{a≠a*} | Q*(s,a*) − Q*(s,a) |     # high = optimal action much better
```
Analytically exact, no trained model. Code: `analysis/claim1/frozen_lake/oracle.py`.

## Metrics
**1. Spearman ρ (primary).** Rank correlation between CCE score and oracle importance over
all non-terminal states. Scale-invariant, outlier-robust, one number + p-value. Report at
training stages (untrained / mid / best checkpoint).

**2. Precision@K (secondary).** Of the oracle's top-K% states, how many are also in CCE's top-K%:
```
Precision@K = |top_oracle_K ∩ top_cce_K| / |top_cce_K|     for K ∈ {5, 10, 20}%
random baseline = K
```
Catches "good overall rank but misses the most critical states."

**3. Sampling KL / Pearson (optional — not currently reported in the paper).** Compare the
sampling distributions `p ∝ (score + ε)^β` for CCE vs oracle: report `KL(oracle‖cce)` and
Pearson `r`. Catches scores that are correctly ordered but too uniform to actually prioritize.

## Sample-size note
If decision points within one episode/game are correlated, the independent unit is the
episode, not the point — size the sample by episodes, not raw positions.

## Pipeline
```
python -m counterfactual_rl.analysis.claim1.frozen_lake.run_analysis
```
Computes oracle + CCE scoring + Spearman + Precision@K + figures. Scored checkpoints are
snapshotted in `paper/repro/cache/checkpoints/`.
