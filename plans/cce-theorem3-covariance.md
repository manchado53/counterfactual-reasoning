# Measure the predicate, not the outcome — Theorems 3, 4 and Corollary 1

Branch: `experiment/theorem3-covariance` · worktree `.claude/worktrees/theorem3-covariance`
Base: master `cdd60ff` (graded-slip merged) + `79b88a5` (suitability + draw-log ported)

## Why

The graded-slip sweep tested Theorem 3 by watching **win rate** as slip rises. It found a knife
edge at slip=0, not the smooth slope the theory wanted. But slip is a **confounded knob**: more
slip dilutes the CCE signal AND makes exploration easier (random slides stumble into the goal),
and the exploration effect dominates. The win-rate curve therefore cannot say anything about
whether the mechanism is real.

The theory is not about win rate. It is about the replay buffer:

    CCE beats PER   <=>   Cov(c,u)/E[c]  >=  Cov(d,u)/E[d]

Both sides are numbers. Measuring them sidesteps the confound entirely — nothing can stumble
into a goal inside a covariance taken over a fixed buffer.

Theorem 3 is a **proved identity**; it cannot be "wrong". What the sweep failed to support is an
unstated bridge assumption — *"environment noise shrinks Cov(c,u) smoothly."* That bridge is
what this tests.

## The idea

FrozenLake 8x8 is small enough to enumerate the whole transition space as the buffer and
brute-force the one quantity nobody usually gets: **true replay utility `u`**.

    53 non-terminal states x 4 actions x 3 outcomes = 636 transitions

**636, not 212.** Under slip, one (s,a) has three outcomes with different next states, different
realized TD, different utility — and the buffer stores them separately. The noise split below is
impossible without that granularity. At slip=0 two slots carry probability 0 and drop out.

Per transition, at a fixed checkpoint and slip level:

- `c_i` — CCE score. Reuse `suitability/rollout_sweep.compute_return_tensor` for the full
  (S,A,N) tensor, then `analysis/metrics.compute_consequence_metric` per action.
- `d_i` — TD error from the Q-net and `env.P`. Reuse `compute_abs_td_per_state`, generalized to
  all actions, with one fix: bootstrap off the **target** net, since that is what training uses.
- `u_i` — **measured, not proxied**. How much does replaying this one transition actually move
  Q toward exact Q*? `Q*` comes from value iteration on `env.P`
  (`suitability/envs.qstar_spread_exact` handles arbitrary slip; `claim1/.../oracle.py` returns
  the full Q* matrix but only for binary is_slippery).

Then evaluate the predicate and sweep slip over the dense sweep's own levels
(0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.133) so the covariance can be overlaid on the known cliff.

## Three theorems for the price of one

The same c, d, u arrays test all three. Only one is testable from the buffer alone.

- **Theorem 3** — which covariance is bigger. A direction. Buffer alone. This is the core figure.
- **Theorem 4** — predicts a *number*, not a sign. But careful: its proof is pure algebra on
  c, d, u, so computing both sides offline puts every point on y=x regardless of the data. That
  validates the code, not the theory. To make it real the left side must come from **actual
  training runs** (realized progress under p^cδ vs p^δ) and only the right side from the buffer.
  Note also it uses the **δ-weighted** base measure, not uniform — using uniform there produces
  points off y=x that look like failed theory but are a weighting bug.
- **Corollary 1** — the noise claim, the part the knife edge actually threatens. Needs δ split
  into reducible and irreducible:

      at Q = Q*, EXPECTED TD is zero, but PER-OUTCOME TD is not.
      that residual spread IS the irreducible noise. exact, from env.P.

  Cor 1 also *assumes* `Cov(εn, u) = 0` — with the split in hand that premise is checkable too,
  not just the conclusion. Without this split the noise claim stays asserted rather than tested.

## Landmines

Each of these silently produces a confident wrong answer.

- **Adam destroys `u`.** From a fresh state Adam's first step is `-lr * sign(g)` — gradient
  magnitude cancels exactly (verified: 500x difference in g gives 0.001% difference in update).
  Using the trainer's `_update_step` makes `u` a sign pattern carrying no information. Use plain
  SGD, or the exact directional derivative `<grad E, grad loss_i>`, which also gets all 636 in
  one JVP instead of 636 clones.
- **`u` factors as `δ x G`** (signed TD times how the net propagates the update). So one-step
  utility is partly TD error by construction and PER wins that half for free. Report
  `Cov(c,δ)` and `Cov(c,G)` **separately** — the second is the real question.
- **`score_states.py:69` builds the agent bare** (`FrozenLakeDQN()`, no config) and `load()`
  restores weights but never the env. At any non-default slip it scores the wrong MDP. This is
  the bug that cost the DoorKey runs; it is dormant only because no FL checkpoint has used a
  custom slip yet, and this experiment is the first thing that will. Always construct from the
  checkpoint's own saved config and assert `abs(env_slip - manifest_slip) < 1e-9` on every load.
- **`score_states.py` collapses per-(s,a) to per-state.** The full tensor exists inside it and is
  discarded. Use `compute_return_tensor` instead.
- ~~**Graded-slip checkpoints are gone.**~~ **WRONG — they exist.** 798 run dirs with 22,725
  `.pkl` files survive in the graded-slip *worktree*
  (`.claude/worktrees/graded-slip-frozenlake/.../frozen_lake/runs/`), roughly 40 checkpoints per
  run, across **11** slip levels (0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.133, 0.166, 0.333, 0.5,
  0.666) x {dqn-uniform, dqn, consequence-dqn}. The sweep ran *from* that worktree so its outputs
  landed there, not under the main repo path. Each run's `metrics.log` header records its own
  `slip_prob`, so runs are self-describing. **No retrain is needed.** Read them in place — `/home`
  is full — and pin the worktree before building on it.
- **The saved config does not survive a bare load.** `load()` restores weights but never rebuilds
  the env, and `score_states.py` constructed `FrozenLakeDQN()` with no config, so a graded-slip
  checkpoint was silently scored in the default slippery MDP — no error, plausible numbers. Fixed
  in `181390d`: use `FrozenLakeDQN.from_checkpoint(path)`, and `load()` now raises on a slip or
  map mismatch (`strict_env=False` is the explicit escape hatch).
- **dqn-uniform diverges at slip 0.** 16 of 30 seeds collapse to 0% win rate while |Q| blows up to
  2513. Diverged runs never early-stop, so they hold the *most* checkpoints and any
  fraction-based selection samples them preferentially — that produced `mean|Q-Q*| = 178` and a
  covariance ratio of `+2.5e5` before the guard went in. Skip runs that never reach a nonzero win
  rate; flag records above `mean|Q-Q*| > 5`.
- **The theorems describe a sampler we do not ship.** Eq 5 defines `p^c ∝ c`; the buffer computes
  `(c + 0.01)^0.25`. Report the predicate under both measures — they agree so far, but they are
  not the same measure, and the exponent is what makes the deployed one near-uniform.

## Order of work

Cheapest first. Each of the first three can kill the idea before any GPU is spent.

1. ~~**Free check**~~ — **DONE (`f462126`). Answer: yes, near-uniform.** `(score + 0.01)^0.25`
   caps the spread at `((1+eps)/eps)^0.25 = 3.17x` — arithmetic, independent of the data. Measured
   across 11 slip levels x 3 seeds x 2 aggregations (66 records): ESS never below **79% of
   uniform**, Gini 0.04–0.24. The graded-slip win-rate cliff was produced by a sampler running
   close to uniform. Second finding: the CCE score is blunt where CCE wins and rich where it does
   not — 2–4 distinct values at slip 0 versus 52–53 at slip 0.666 — and score magnitude is
   bimodal across seeds, since a policy that rarely reaches the goal scores `c ≈ 0` everywhere.
   Figures and data in `docs/figures/theorem3_step1/`.
2. **Build + validate** — compute c, d, u on the three checkpoints already committed at
   `paper/repro/cache/checkpoints/seed_{0,1,2}/`. Exercises every code path with zero cluster
   time. Expect a real edge case: on an untrained net no rollout reaches the goal, every `c` is
   0, and the ratio is 0/0. That is a finding (CCE has no signal until the policy can sometimes
   reach the goal), not a bug — report `frac_c_zero` and skip the predicate there.
3. **Falsify** — build a sampler drawing ∝ measured `u` and race it against uniform in a real
   short run. If it does not win, `u` is not utility and everything downstream is decoration.
   Better found here than in review. Run this BEFORE the sweep.
4. **Gate** — slip {0, 0.333} only. If the covariances do not move between the extremes, the
   five middle levels are wasted GPU.
5. **Sweep** — **11** levels, **no GPU and no retrain**: the checkpoints already exist (see
   landmines). Primary arm is **dqn-uniform**, so neither priority scheme shaped the weights being
   scored — anything else is circular. Index stages by **achieved win rate, not episode**: at slip
   0.10 a net learns faster, so a fixed episode means different competence at different slip,
   which lets the confound back in through the checkpoint. Implemented as
   `select_by_winrate()`; deduplicate targets, because early-stopped runs jump 0 → 1 with nothing
   at 0.5. Apply the divergence guard. Any *fresh* runs still need `early_stop_win_rate: None`
   and `dh-node12` in `--exclude`.
6. **Figures + PROVENANCE**, following the graded-slip template. Write the primary analysis down
   **before** the sweep runs so the crossing point cannot be called cherry-picked.

## Guardrails

- Cache results to `paper/repro/cache/` immediately after the sweep — the last dataset was lost
  to disk pressure, and `**/runs` is gitignored.
- Average `u` over several checkpoints and seeds, not one snapshot; one gradient step is myopic
  and a single reading will not survive review.
- Assert the Theorem 3 identity holds to `< 1e-10` on every record. If it ever fails, the
  weighting is wrong.
- `--dry-run` before any sbatch.

## Not in scope

Another win-rate sweep. The reward-noise environment (the deconfounded Theorem 3 test) — that is
the follow-up, and only worth building if the bridge survives this. Merging
`research/cce-buffer-diagnosis` to master (separate job: rename plus guard, not a blind merge).

## Related, unblocked, and not this branch's job

The paper's Claim-1/Claim-2 contradiction (C1 measured on slippery FL, C2 null explained by
slippery FL diluting the signal) is still open and is the first thing a reviewer will find.
Also free: `BLARBLARBLAR` in section IV.D, two broken `Algorithm ??` refs, the 25-vs-10 seed
mismatch, SMAX PER quoted as 0.68 against a verified 0.710, and a placeholder author email.

One more for Jeremy: config says `consequence_aggregation: weighted_mean`, but `action_probs` is
never passed, so `compute_consequence_metric` falls through to **max**. Theorem 1 requires the
aggregation to be monotonic *and linear*; max is monotonic but not linear.
