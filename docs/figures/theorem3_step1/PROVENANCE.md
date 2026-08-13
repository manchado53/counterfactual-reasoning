# Theorem-3 step 1 — priority flatness: provenance & reproduction

Experiment: **plan step 1 of `plans/cce-theorem3-covariance.md`** — the free check
that runs before any covariance measurement. Question: is the deployed replay
sampler concentrated at all? If it sits near uniform, the graded-slip sweep
tested near-uniform replay rather than Theorem 3.

## The finding

The buffer shapes priorities as `p = (score + eps)^beta` with `eps=0.01`,
`beta=0.25` (`agents/shared/consequence_buffers.py:125`). For a score bounded in
[0, 1] that caps the spread between any two transitions at

    ((1 + 0.01) / 0.01)^0.25 = 3.17x

independent of the data. Measured across 11 slip levels x 3 seeds x 2
aggregations (66 rows), the realised effective sample size never falls below
**79% of uniform**, and Gini stays in 0.04-0.24.

Second observation: the CCE score is **blunt at low slip and rich at high slip** —
2-4 distinct values at slip 0 versus 52-53 at slip 0.666. The signal is coarsest
in the environment where CCE's Claim-2 win is largest. Score magnitude is also
strongly bimodal across seeds at the same slip (e.g. slip 0.10: `c_mean` 0.035,
0.051, 0.342), consistent with the plan's predicted edge case — a policy that
rarely reaches the goal yields `c ≈ 0` everywhere.

## Data sources

| Artifact | Path |
|---|---|
| Analysis + figure code | `analysis/theorem3/priority_flatness.py` |
| Summary rows (66) | `step1_ess.json` (this directory) |
| Per-state scores at both extremes | `hero_scores.npz` (this directory) |
| Source checkpoints (NOT committed) | graded-slip worktree, `agents/frozen_lake/runs/<job_id>/checkpoints/` |

Checkpoints are the **last** checkpoint of `consequence-dqn` runs from the
graded-slip sweep (merged at `cdd60ff`). Runs are located by their own
`metrics.log` header (`slip_prob`, `algorithm`, `seed`). Override the search root
with `GRADED_SLIP_RUNS=<path>`.

## Figure

| File | What it shows |
|---|---|
| `fig_priority_flatness.png` | Top: realised sampling probability per state relative to uniform, at both slip extremes, against the 3.17x structural ceiling. Bottom left: ESS as % of uniform, every run and slip level. Bottom right: distinct CCE score values vs slip. |

## Reproduce

    python -m counterfactual_rl.analysis.theorem3.priority_flatness            # measure + figure
    python -m counterfactual_rl.analysis.theorem3.priority_flatness --figure   # figure from cache

Analysis-side only; no cluster time. The measurement pass takes a few minutes on
CPU (rollouts at `cf_horizon=200`, 20 rollouts per state-action).

## Caveats

- **CCE component only.** The deployed combined priority also carries a TD term.
  Under multiplicative mixing the joint ceiling is `3.17^2 = 10.05x`. Measuring
  the realised joint spread needs the 636-transition enumeration (step 2).
- **ESS is over the 53 distinct non-terminal states**, not the live buffer's
  visitation-weighted composition. The 3.17x ceiling is unaffected by this; the
  ESS figure is indicative rather than the buffer's true ESS.
- **Last checkpoint per run only** — no view of how flatness evolves during
  training.
- Both aggregations are reported. `weighted_mean` under a greedy policy puts zero
  weight on every alternative, so `compute_consequence_metric` falls through to a
  plain mean; the `mean` rows are what `weighted_mean` yields. See GitHub issue #3
  for the `max` fallback itself.
