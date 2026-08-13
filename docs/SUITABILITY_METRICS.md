# CCE Suitability — Metrics Cookbook

> **How we forecast & debug "will CCE help in this environment?"** Definitions, formulas, and
> how-to-compute only. **No results or numbers here** — current results live in `/lab-notebook.md`
> (status). These are **decision tools for us** (what to pursue / avoid), not (yet) paper figures.

## What this measures
CCE prioritizes replay by how much the *action choice* changed the outcome. It only helps when the
environment actually has decisive choices, the signal is visible over the noise, CCE's ranking
differs from plain TD-error, and the high-stakes states are revisited. These metrics — all cheap,
from policy rollouts — score each of those conditions, so we can predict win/no-gain **before**
spending compute and diagnose **which** condition broke when CCE underperforms.

## Organizing theory — worth replaying = GAIN × NEED
From Mattar & Daw 2018 (Nature Neuro): the value of replaying a transition factors as
`EVB = GAIN × NEED`. **GAIN** = how much fixing this state changes the policy (≈ the action gap).
**NEED** = how often the policy revisits the state (discounted occupancy). **CCE estimates GAIN
only.** So the metrics below are *conditions on GAIN* (is it real, visible, distinct, faithful,
reachable) plus the *NEED* term CCE ignores.

## The atom — the rollout tensor and per-state stakes
Roll the loaded (greedy) policy out. For each scored state `s`, for each action `a`, run `N`
rollouts that take `a` then follow the greedy policy to the horizon → discounted returns.
```
returns(S, A, N)                      # the one tensor everything reuses
m(s,a)   = mean_n returns[s,a,:]      # (S, A)  mean return per (state, action)
C(s)     = max_a m(s,a) − min_a m(s,a)   # (S,)  the STAKES of state s (spread over actions)
```
`C(s)` big = a fork (the move decides the outcome); `C(s)` ≈ 0 = a throwaway state.
Reuse: `FrozenLakeConsequenceDQN._build_rollout_fn` (consequence_dqn.py) →
`agent._compiled_rollout_fn(params, states, ALL_ACTIONS, keys) -> returns(B,A,N)`; mirror the call
loop in `analysis/diagnostics/compute_diagnostics_fl.py` but keep the full tensor.

---

## The v1 metrics
Each: formula · what it DECIDES · pass-when · code/reuse. (Status: **v1** = built now.)

### 1. Concentration — "are there forks to prioritize?"  · *forecast*
```
Concentration = gini( { C(s) : s scored } )      # + top-k mass, entropy as sub-stats
```
Lumpy (high Gini) = a few decisive states → CCE has a target. Flat → nothing to prioritize.
Pass when: Gini high (≳ 0.5) / top-k mass concentrated. Reuse: `plot_diagnostics.py:gini`.

### 2. SNR (greedy) — "is the fork louder than the noise?"  · *forecast*
```
between(s) = Var_a( m(s,a) )            = returns.mean(2).var(1)   # signal: action effect
within(s)  = mean_a( Var_n returns[s,a,:] ) = returns.var(2).mean(1)  # noise: env stochasticity
SNR        = median_s [ between(s) / within(s) ]
```
Rollouts continue **greedily** after the forced action, so `within` is *environment* noise only,
not exploration noise (≈ noise-normalized action gap; Bellemare 2016). Det env → `within ≈ 0` →
SNR huge; stochastic env → noise drowns signal → SNR ≈ 1. Pass when: SNR ≳ 3.
Reuse: falls straight out of the atom tensor (two numpy lines, no extra rollouts). Guard `within==0`
with eps and report whether it was exactly 0.

### 3. DISTINCT-TD — "does CCE rank moves differently from PER?"  · *GO / NO-GO*
```
DISTINCT-TD = 1 − | Spearman( cce_priority , |TD| ) |
|TD(s)| = | r + γ max_a' Q(s',a') − Q(s,a_greedy) |       # greedy-action TD per state
```
If CCE ranks like TD-error, it is just a slow copy of PER and **cannot beat it** — this is the
single most important check. `|ρ|` (not ρ) because a perfect *reverse* of TD is still "just TD,
flipped" = no new information; we want ρ ≈ 0 (a shapeless cloud). Pass when: DISTINCT-TD ≳ 0.5.
Reuse: `compute_consequence_metric` for CCE; `compute_diagnostics_fl.py:build_td_fn` for |TD|;
`plot_diagnostics.py:spearman_boot`. **CCE and |TD| must use the SAME (greedy) action at `s`.**

### 4. GAIN-fidelity — "is CCE's guess correct?"  · *calibration (FrozenLake only)*
```
GAIN-fidelity = Spearman( cce_priority , qstar_spread )
qstar_spread(s) = max_a Q*(s,a) − min_a Q*(s,a)          # exact, from value iteration
```
The honesty check: does the cheap rollout stake match the exact answer? Only computable where we can
solve the env (FrozenLake). **Compare against `qstar_spread`, NEVER against `C(s)` itself — that is
circular.** Passing here licenses trusting the cheap metrics on envs without an oracle. Pass when:
Spearman high (≳ 0.5, ideally ≳ 0.7). Reuse: `analysis/diagnostics/value_iteration.py:compute_qstar`,
`stakes_from_qstar`.

### 5. NEED — "do the high-stakes states get visited?"  · *forecast*
```
NEED = Spearman( C(s) , d(s) )
d(s) = discounted visit frequency under the greedy policy
       v1 (tabular):  Σ_t γ^t · 1[S_t = s] over rollouts, normalized
       exact (FL opt-in):  row of successor representation (I − γ T_π)^-1
```
CCE forgets NEED; a huge fork in a never-revisited dead-end is wasted replay. Pass when: high-stakes
states have high occupancy. **Caution:** under a near-random (early) checkpoint, `d(s)` is
degenerate and NEED is noisy — report `coverage` (fraction of scored states with `d(s)>0`) and use
Spearman (rank). Reuse: greedy rollouts via `agent._greedy_action` + `env.step`; exact via `env.P`.

### 6. HORIZON-fit — "is the rollout long enough to see the payoff?"  · *gate*
```
HORIZON-fit = cf_horizon / effective_horizon
v1 proxy: effective_horizon = smallest H where mean_s C(s) stabilizes (rel. change < 5%),
          measured by re-running rollouts at H ∈ {2,4,8,16,32} on a state subset.
```
If `cf_horizon` is shorter than the credit-assignment length, rollouts are cut off before the reward
and `C(s)` is under-estimated → silent CCE failure. Pass when: ratio ≳ 0.8. v1 uses the proxy
(flagged `is_proxy=True`); the principled version is Laidlaw et al. 2023 effective horizon.

---

## The FrozenLake bridge (calibration step)
Before trusting any cheap metric on Connect Four / SMAX, confirm on FrozenLake that the rollout
estimate is honest: `Spearman( C(s) from rollouts , qstar_spread )` should be high (this *is*
GAIN-fidelity). Once the bridge passes, metrics 1–3,5,6 are trusted blind on envs without an oracle.

## Policy / warmup-sweep note
Every metric depends on which policy `π` is rolled out. Measure across a **warmup sweep** of
checkpoints (random / ~10% / ~30% / full trained) and report drift, rather than a single snapshot.
Early-checkpoint NEED and stochastic-env C(s) are expected to be noisy (report CIs / coverage).

---

## Deferred (Option B — needs trainer instrumentation)
Not computed in v1: these need the **realized replay sampling distribution** logged during a real
training run (the trainer doesn't record which transitions it draws). Formulas kept here so they're
ready to wire up. (Status: **deferred**.)

### precision@k — "are the drilled transitions the RIGHT ones?"  *(primary sanity)*
```
precision@k = | drilled_top-k ∩ true-stakes_top-k | / k
drilled_top-k = the k most-sampled transitions during a run
true-stakes_top-k = top-k by C(s) (or exact Q* spread where available)
```
The real quality measure of CCE's focus. Resume: instrument the trainer to log sampled indices,
then add `compute_realized_sampling()` to `rollout_sweep.py`.

### ESS — "did the draws collapse?"  *(collapse alarm only)*
```
ESS = 1 / Σ_i p_i^2        # p = realized sampling distribution over the buffer
```
ESS near 1 = collapsed onto a handful of transitions (lost coverage). **Alarm only** — CCE is
*supposed* to be spiky, so low ESS is not a fault by itself; it just flags degenerate collapse.

---

## Pipeline
```
python -m counterfactual_rl.analysis.suitability.run_suitability \
    --run-id <existing_FL_run> --envs FL-det FL-stoch --out scorecard.json --fig scorecard.png
```
Computes the atom tensor per warmup checkpoint, then all v1 metrics, emits `scorecard.json` + a
figure, and can inject real numbers into `docs/figures/mock_preview/dashboard.html`. Smoke test:
`python -m counterfactual_rl.analysis.suitability._smoke` (asserts the bridge + SNR det≫stoch).
Code: `src/counterfactual_rl/analysis/suitability/` (envs, rollout_sweep, metrics, scorecard, run).
