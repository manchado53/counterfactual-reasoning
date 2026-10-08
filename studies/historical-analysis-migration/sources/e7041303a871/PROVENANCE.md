# Theorem-3 covariance experiment — provenance & reproduction

Experiment: `plans/cce-theorem3-covariance.md`. Theorem 3 says CCE priority beats
TD-error priority exactly when `Cov(c,u)/E[c] >= Cov(d,u)/E[d]`. Both sides are
computable numbers, so the question can be settled directly instead of inferred
from win rates — which is a confounded measurement, because slip both dilutes the
CCE signal and makes exploration easier.

## Headline result

Across all 11 graded-slip levels, on neutral DQN-uniform checkpoints matched by
achieved win rate:

- **CCE wins the predicate in 12 of 89 converged runs.**
- **The CCE side's median is not positive at any of the 11 slip levels.** At slip 0
  it is exactly 0.00; everywhere else it is negative.
- **The TD side's median is positive at all 11 levels.**
- Deterministic (slip 0) is the only level where CCE gets meaningful traction:
  5/13, versus 0/10 at full slip. Fisher one-sided p = 0.038 (theory measure),
  p = 0.017 (deployed measure).

The plan framed this as a test of an unstated bridge assumption — *"environment
noise shrinks Cov(c,u) smoothly."* The measured answer is that Cov(c,u) is at or
below zero across the whole axis, so there is no span for noise to shrink. This
supports the Claim-2 stochastic null more strongly than it supports the
Claim-2 deterministic win.

**Load-bearing caveat:** all of the above assumes `u` deserves the name "replay
utility". The falsification race that tests this (step 3) is a separate
experiment; until it passes, these covariances are numbers of unproven meaning.

## Step 1 — priority flatness

The buffer shapes priorities as `p = (score + eps)^beta`, `eps=0.01`, `beta=0.25`
(`agents/shared/consequence_buffers.py:125`). For a score in [0,1] that caps the
spread between any two transitions at `((1+eps)/eps)^beta = 3.17x`, independent of
the data. Measured across 11 slip levels x 3 seeds x 2 aggregations (66 records),
effective sample size never falls below **79% of uniform**; Gini stays in
0.04–0.24.

Second observation: the CCE score is blunt at low slip and rich at high slip —
2–4 distinct values at slip 0 versus 52–53 at slip 0.666 — and magnitude is
bimodal across seeds, consistent with a policy that rarely reaches the goal
scoring `c ≈ 0` everywhere.

## Step 2 — c, d, u over the transition space

FrozenLake 8x8 is enumerated as the buffer: 53 non-terminal states x 4 actions x
3 outcomes = 636 transitions (212 at slip 0, where two outcome slots carry zero
probability).

| Quantity | How it is obtained |
|---|---|
| `c` | CCE score of (s,a) from policy rollouts, under **both** `max` and `mean` aggregation |
| `d` | \|TD error\|, bootstrapped off the **target** net, because that is what training uses |
| `u` | `delta * G` where `G = -2<grad E, grad Q(s,a)>` and `E = mean\|Q - Q*\|`; `Q*` from value iteration on `env.P` |

`u` is the exact directional derivative rather than a finite optimiser step.
Adam's first step from a fresh state is `-lr * sign(g)`, which cancels gradient
magnitude and would reduce `u` to an uninformative sign pattern; the derivative
also yields all 636 transitions in one JVP instead of 636 network clones.
Validated against hand-rolled SGD steps: max relative error 1.7e-03 to 4.3e-02.

Theorem 3 and Theorem 4 are algebraic identities in `c, d, u`, so computing both
sides validates the code rather than the theory. All **82** records pass; worst
residual **1.07e-14**.

## Data sources

| Artifact | Path |
|---|---|
| Step 1 analysis + figures | `analysis/theorem3/priority_flatness.py` |
| Predicate measurement | `analysis/theorem3/predicate.py` |
| Result figures | `analysis/theorem3/figure_predicate.py` |
| Step 3 falsification race | `analysis/theorem3/utility_sampler.py` |
| Step 1 summary (66 rows) | `step1_ess.json` |
| Repro-checkpoint predicate (9 rows) | `step2_predicate.json` |
| Per-slip predicate | `step2_graded_slip<p>_dqn-uniform.json` (11 files) |
| Per-state scores at both extremes | `hero_scores.npz` |

Source checkpoints are **not** committed. They are the graded-slip sweep's own
run dirs, in the sibling worktree
`.claude/worktrees/graded-slip-frozenlake/.../frozen_lake/runs/<job_id>/checkpoints/`
— 798 runs, 22,725 `.pkl` files. Runs are located by their own `metrics.log`
header (`slip_prob`, `algorithm`, `seed`). Override the search root with
`GRADED_SLIP_RUNS=<path>`.

## Figures

| File | What it shows |
|---|---|
| `fig_priority_flatness.png` | Realised sampling probability per state vs uniform, ESS across all runs and slip levels, and distinct CCE values vs slip |
| `fig_beta_ceiling.png` | The 3.17x structural ceiling as a function of beta, and real scores pushed through two exponents |
| `fig_predicate_by_env.png` | Per-run predicate margin, deterministic vs full slip |
| `fig_predicate_slip_axis.png` | Win fraction and both medians across all 11 slip levels |

## Reproduce

    python -m counterfactual_rl.analysis.theorem3.priority_flatness
    python -m counterfactual_rl.analysis.theorem3.predicate
    python -m counterfactual_rl.analysis.theorem3.predicate --graded --slip 0.0 --algo dqn-uniform --n-seeds 25 --fracs 0.0 1.0
    python -m counterfactual_rl.analysis.theorem3.figure_predicate

Analysis-side only; no cluster time and no retrain. Step 3 is the exception and
submits SLURM jobs (`--submit --dry-run` first).

## Two guards that changed the answer

- **Wrong-MDP scoring.** `load()` restores weights but never rebuilds the env, and
  `score_states.py` built the agent with no config, so a graded-slip checkpoint was
  silently scored in the default slippery MDP — no error, plausible numbers. Fixed
  in `181390d`: `FrozenLakeDQN.from_checkpoint()`, plus an assert on slip and map.
- **Divergence.** DQN-uniform on deterministic FL diverges in 16 of 30 seeds: win
  rate collapses to 0 while |Q| reaches 2513. Diverged runs never early-stop, so
  they hold the most checkpoints and fraction-based selection sampled them
  preferentially — producing `mean|Q-Q*| = 178` and a covariance ratio of `+2.5e5`.
  Runs that never reach a nonzero win rate are skipped; records above
  `mean|Q-Q*| > 5` are flagged.

## Caveats

- **`u` is unvalidated** until step 3 completes. This is the load-bearing one.
- `u` is myopic: one gradient step, first order. Real replay utility compounds.
- The predicate is evaluated with the enumerated transition space weighted
  uniformly. A live buffer is visitation-weighted, a different measure.
- ESS in step 1 is over distinct non-terminal states, not buffer composition. The
  3.17x ceiling is unaffected.
- One checkpoint per seed. Usable counts are thin at low slip (3–7 runs at
  0.02–0.10) because of divergence.
- **The theorems describe a sampler we do not ship.** Eq 5 defines `p^c ∝ c`; the
  buffer computes `(c + 0.01)^0.25`. Both are reported; they agree here.
- `weighted_mean` aggregation silently degrades to `max` on FrozenLake — see
  GitHub issue #3. Every result is reported under both, and none depend on it.

## Retracted

An earlier pass, on 7 and 8 records, reported that *both sides of the predicate
swap sign between environments*. With 13 and 10 records the TD side's median in
deterministic FL flips from −1.5e−02 to +1.0e−02. That claim was a small-sample
artifact and has been withdrawn; the deterministic win fraction also fell from
4/7 to 5/13.
