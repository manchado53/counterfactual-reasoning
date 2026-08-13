# Theorem 3 covariance experiment — measure the predicate, not the outcome

## Why

The graded-slip sweep (merged, `cdd60ff`) tested Theorem 3 by watching **win rate** as slip
rises. It found a knife edge at slip=0, not the smooth slope the theory wanted. But slip is a
**confounded knob**: more slip dilutes the CCE signal AND makes exploration easier (random
slides stumble into the goal). The exploration effect dominates, so the win-rate curve cannot
tell us whether Theorem 3's mechanism is real.

Theorem 3 is not a statement about win rate. It is a statement about the replay buffer:

    CCE beats PER   <=>   Cov(c,u)/E[c]  >=  Cov(d,u)/E[d]

Both sides are numbers we can compute directly. Measuring them sidesteps the confound
entirely — no exploration effect can contaminate a covariance taken over a fixed buffer.

Note: Theorem 3 is a **proved identity**. It cannot be "wrong". What the sweep actually failed
to support is an unstated bridge assumption — *"environment noise shrinks Cov(c,u) smoothly."*
This experiment tests that bridge, which is the thing that needs revising.

## The idea

FrozenLake 8x8 is tiny: 53 non-terminal states x 4 actions = 212 transitions. Small enough to
enumerate the entire transition space as the buffer and brute-force the one quantity nobody
usually gets to see: **true replay utility `u`**.

For a fixed policy checkpoint at slip level p, per transition i:

- `c_i` — CCE score, from existing rollout scoring (`claim1/frozen_lake/score_states.py`)
- `d_i` — TD error, from the Q-net and `env.P`
- `u_i` — **measured, not proxied**: clone the network, take one gradient step on transition i
  alone, record how much global error `mean|Q - Q*|` drops. 212 probes per checkpoint, cheap.

Exact `Q*` already exists via value iteration on `env.P` (`claim1/frozen_lake/oracle.py`).

Then compute both sides of the predicate, plus the direct check `E_{p^c}[u]` vs `E_{p^d}[u]`,
and sweep across the same slip levels the dense sweep used (0, 0.02, 0.04, 0.06, 0.08, 0.10,
0.133) so the covariance ratio can be overlaid on the win-rate cliff.

## What we learn

- **Ratio flips near slip 0.02, where the win dies** -> Theorem 3's mechanism is real; the flat
  win-rate curve was the exploration confound hiding it. The paper gets much stronger, and the
  slip sweep becomes supporting evidence instead of a negative result.
- **Ratio decays smoothly while the win cliffs** -> the bridge assumption is genuinely wrong.
  Also publishable, and it tells Jeremy exactly which lemma is missing.

Either way we stop guessing from win rates.

## Scope

Analysis-side work plus a small retrain to get checkpoints per slip level (the graded-slip runs
did not preserve usable per-level checkpoints). ~21 short FrozenLake runs, a few GPU-hours —
trivial next to the 250 GPU-h the dense sweep spent.

Explicitly NOT in scope: another win-rate sweep, the reward-noise environment (that is the
follow-up, and only worth building if the bridge survives this test).
