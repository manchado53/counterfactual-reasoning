# Graded Stochasticity Sweep — FrozenLake

**Issue:** manchado53/counterfactual-reasoning#1
**Branch:** experiment/graded-slip-frozenlake
**Source:** 07/30 meeting with Jeremy, action item #1

## What we're doing

Right now we have two dots: deterministic FrozenLake (CCE-mul crushes PER) and full-slip
FrozenLake (everyone ties, a null). Theorem 3 says CCE's edge over PER should grow smoothly
as environment noise shrinks. We're going to fill in the curve between the two dots — run the
same experiment at several slip levels and show the edge rise as slip falls.

## Why it matters

It converts the stochastic null from an awkward hole in the paper into a *confirmation* of the
theory's prediction. "We predicted the slope, then measured it." Strong story for a top venue.

## The one code change

`envs/frozen_lake.py` only knows binary slippery (1/3 each) vs deterministic. The 3-outcome
table `[(a-1)%4, a, (a+1)%4]` is already shaped for graded slip — we add a `slip_prob = p` knob
and set the outcome probabilities to `[p/2, 1-p, p/2]`:
- p = 0    -> deterministic
- p = 2/3  -> today's full slip
`is_slippery=True` keeps working (maps to p=2/3). The Claim-1 oracle reads `env.P`, so `P` must
carry the same graded probabilities — value iteration then stays correct at every slip level.

## The sweep

- Slip levels p: {0.0, 0.166, 0.333, 0.5, 0.666}  (5)
- Algorithms: DQN-uniform, DQN+PER, CCE-only, CCE+TD (mul)  (additive dropped per 07/30)
- Seeds: 10 to start; bump to 25 on any level where the trend is noisy
- Metrics (rliable): final IQM win rate + P(CCE-mul > PER) per level
- Headline plot: advantage = IQM(CCE-mul) - IQM(PER) vs slip -> expect monotone

## Guardrails (before any big run)

- Unit test: p=0 matches deterministic, p=2/3 matches old slippery, rows sum to 1.
- Sanity: plain DQN must solve each slip level before we trust CCE numbers.
- Always --dry-run the sweep first. Watch the .out with a Monitor after sbatch.
- Pre-flight: total_variation metric, cf_horizon = full episode, crossing logic for eval/save.
