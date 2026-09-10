# JaxNav: buy replay coverage by selling redundant rollouts

Branch `research/jaxnav-greedy-coverage`, cut from `research/buffer-instrumentation`
(needs the `scored_frac` / `rollout_*_frac` instrumentation that only exists there).

## The idea in one line

Keep the buffer at its full 100,000 and raise the share of it that carries a *measured*
CCE score, by making each score ~20x cheaper instead of by throwing data away.

## Why now

Coverage has only ever been raised on JaxNav by SHRINKING the buffer. All 32 such runs
collapse (~10% win rate, p=0.001). That confounds "more coverage" with "less data", so
the coverage question itself was never actually answered at full capacity. Adrian
pushed on exactly this and he was right: `interval4x` and `sample4x` exist only at
25,000 episodes with 2 seeds, inside the epsilon decay, as a mechanism probe. Never an
outcome test.

Against Jeremy's five levers:

```
 #  lever                config key         moves coverage?   our evidence
 1  CCE batch size       n_score_sample     YES               untested at outcome  <- PULL
 2  CCE update frequency score_interval     YES               untested at capacity
 3  CCE rollout length   cf_horizon         no                goal-rate flat H=60..160
 4  CCE num rollouts     cf_n_rollouts      no                R=40 was WORSE than R=20
 5  Buffer size          buffer_capacity    YES               32 runs, all collapse. DEAD
```

Levers 3 and 4 set the price of a score, not how many transitions get one. That is the
door: **sell lever 4 to buy lever 1, for zero extra GPU.**

```
  scoring work = n_score x actions x rollouts x horizon
     today       64 x 15 x 20 x 60 = 1,152,000
     greedy_cov  1,280 x 15 x 1 x 60 = 1,152,000     identical
```

Selling lever 4 is free because JaxNav's transitions are deterministic: at
`cf_rollout_temperature=0` the 20 rollouts of an action are the SAME rollout 20 times.
One rollout is exact, not an approximation.

## Forced consequences of R=1 (not free choices)

- `consequence_metric` must be **wasserstein**. `compute_total_variation` returns exactly
  0.0 when a sample has `len < 2`, so TV + R=1 silently zeroes every score and degrades
  CCE to uniform with no error.
- At `cf_rollout_temperature=0` the action-probability softmax (`consequence_dqn.py:243`,
  `temp = max(cf_rollout_temperature, 1e-3)`) collapses to one-hot, so
  `weighted_mean` silently BECOMES `max`. Measured: effective sample size 1.0/14 vs
  11.8-14.0/14 at T=0.5. To run weighted_mean at T=0 we add a separate
  `cf_action_prob_temperature` knob (see below).

## Code change (jax_nav only, ~4 lines)

`agents/jax_nav/consequence_dqn.py:243` gains an optional `cf_action_prob_temperature`.
Unset -> falls back to `cf_rollout_temperature`, i.e. byte-identical to today. `max`
aggregation ignores action_probs entirely and cannot be affected. No other env touched.
Parity-tested before launch.

## Arms — 5 seeds each, cell 8x8 fill 0.3, 250,000 episodes

Shared: `algorithm=consequence-dqn`, `priority_mixing=multiplicative`,
`mu_c=mu_delta=1.0`, `consequence_metric=wasserstein`, `cf_rollout_temperature=0`,
`cf_n_rollouts=1`, `buffer_capacity=100_000`, `score_interval=250`, `cf_gamma=0.99`.

```
  arm               H    bootstrap  n_score  coverage   aggregation      h/job
  greedy_base_max   60      ON          64    0.86%     max                1.6
  greedy_base_wm    60      ON          64    0.86%     weighted_mean      1.6
  greedy_cov_max    60      ON       1,280   ~17%       max                8.6
  greedy_cov_wm     60      ON       1,280   ~17%       weighted_mean      8.6
  ------------------------------------------------------------------ 20 jobs, ~102 GPU-h

  SMOKE-GATED, launch only if the smoke passes:
  greedy_full_max  200      OFF        384   ~5.2%      max                8.6
  greedy_full_wm   200      OFF        384   ~5.2%      weighted_mean      8.6
  ------------------------------------------------------------------ 10 jobs, ~86 GPU-h
```

`greedy_cov_* vs greedy_base_*` differs by **one key**, `n_score_sample`. That is the
experiment. The aggregation row is insurance against one of the two being degenerate at
R=1, not a powered comparison — the effect is ~3 pp and 5 seeds resolves ~35 pp.

**bootstrap ON for the H=60 arms.** Matched on mixing+aggregation at fill 0.3, 250k:
multiplicative arms collapse (<30% final) in 7/24 seeds without bootstrap and 0/13 with
it (Fisher p=0.038); means +11.7 and +16.2 pp, both n.s. Caveat: a similar JaxNav
reliability claim was retracted 2026-08-15 (lab-notebook:622). Treat this as a knob
setting, not a reportable result.

`greedy_full` drops bootstrap because at H=200 = `max_steps` every rollout terminates,
so there is nothing to bootstrap from. Its premise is doubted by our own measurement:
goal-reach rate was 8% at H=20, 12% at H=60, flat through H=160 (lab-notebook:624).
Hence the gate.

## Baselines — reuse from disk, do not rerun

At fill 0.3, 250k: `uniform` n=26, `per` n=18, `cce_mul_weighted_mean` n=16,
`cce_mul_max` n=8, plus the bootstrap variants. Seed NUMBERS do not need to match:
arms share one RNG stream and CCE consumes extra keys for rollouts, so seed s is a
different world in each arm. Measured cross-arm correlation over 12 pairs at n=25 is
scattered around zero. **Analysis must be UNPAIRED (Mann-Whitney), not paired.**

## Smoke gate — 6 jobs, 1 seed, 5,000 episodes, ~15 min

```
  scored_frac          greedy_cov_*    ~17%   (derived; 1.6% derivation reads ~2x high)
  scored_frac          greedy_base_*   ~0.86%
  rollout_alive_frac   greedy_full_*    ~0.0%  <- the H=200 guarantee. if not, KILL the arm
  rollout_goal_frac    greedy_full_*    > 9.2% <- must beat the H=60 baseline, or KILL
  score_n_zero/total   all             NOT ~100%
  score_hist           all             graded, not two spikes
  no OOM, no new errors in .out
```

## Readout

Final-20-eval mean win rate per seed. Unpaired Mann-Whitney plus effect size and CI.
**No significance claim at n=5** — paired sd is 28 pp, so 5 seeds resolves ~35 pp and the
gaps of interest are ~10 pp. The deliverable is a measured coverage number, a direction,
and a collapse-or-not.

## Explicitly not doing

Buffer capacity (dead, 32 runs). Lever 2 / score_interval as a separate arm (equivalent
to lever 1 for coverage; deferred). The regret-style "compare against the best
alternative only" signal (new idea, needs code, deserves more than 5 seeds — file as an
issue). Fixing FrozenLake's missing `action_probs` before the ICLR deadline: the 208-run
FL win is a win under `max`, so relabel it honestly rather than re-run everything.
