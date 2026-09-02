# Bootstrap the CCE counterfactual rollout (issue #7)

## Why

The CCE score compares, for each action, the distribution of returns from short rollouts. Right
now a rollout that simply *runs out of horizon* is scored as if it had died with nothing: the
code returns the accumulated reward and stops. Under a sparse goal-only reward that means a
hard zero.

So the estimator conflates two different things:

    "you were still going when I stopped watching"   ->  0
    "you crashed and it's over"                      ->  0

That is a genuine bug, not a tuning problem. A truncated rollout should carry the value of
wherever it stopped. That is just an n-step return.

Measured consequence on JaxNav (probe 274242 / 274250): the CCE score is *exactly* zero for
39-88% of buffer states, so replay priority collapses to uniform and no `beta` can rescue it.

## What we're doing

Add an opt-in `cf_bootstrap` flag. When on, a rollout that ends because the horizon ran out gets
`+ gamma^H * max_a Q_target(s_H, a)` added to its return. Rollouts that reached a real terminal
(goal, crash, timeout) are untouched — for those, zero is the correct value.

The target network, not the online one, so the score isn't chasing the same moving estimate the
TD update is.

Default is **off**, so every existing run and figure stays reproducible.

## Scope

**JaxNav only.** That is where the problem is measured and where we can verify in two minutes
against saved checkpoints.

No shared abstraction, deliberately. The rollout is written five separate times, once per env,
and unifying the scoring core two weeks before the abstract deadline is the wrong risk — a
silent refactor bug there corrupts every result without failing loudly. Filed as #12 instead.

FrozenLake is also left alone. It has a related but distinct defect (deterministic rollouts, so
its score collapses to two values) and it carries the paper's headline C2 result, so changing
how it scores would invalidate the 25-seed run behind that number. Filed as #13.

## How we'll know it worked

Re-run the existing score probe on the same checkpoints and compare. Prediction, from the
terminal-reason split, is that the zero fraction drops from 80.7% to ~48% on the open 8x8 map and
only from 93.1% to ~80% on the cluttered one.

The number that actually decides it is `ess_frac`, not the zero fraction. A score that becomes
uniformly nonzero prioritises no better than one that is uniformly zero — FrozenLake already
showed that trap (median score 0.667 but `ess_frac` 0.991). If zeros fall and `ess_frac` does
not, the fix is cosmetic and we say so.

## What this does not fix

Crashes are 42-75% of rollouts on JaxNav and they stay at zero, correctly — under our reward a
crash really is worth the same as wandering. Separating those is a reward-design decision, not a
truncation bug. Tracked in issue #11 along with the head-to-head against simply lifting the
horizon.

Only the `ALIVE@H` bucket is truncated, which is 12.7-32.7% of rollouts depending on the cell.
So this is a real correctness fix with a bounded payoff, not a solution to the zero problem.
