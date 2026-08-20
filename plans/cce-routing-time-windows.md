# Option A — make routing have cliffs (time windows + strandable shift)

**Branch:** research/cce-cvrp-logistics · **Status:** plan, awaiting sign-off
**Supersedes nothing.** Budget mode stays; this adds two flags on top of it.

## Why

Budget routing gave a clean Claim-2 null over ~2,000 runs. The cause is measured, not guessed:
routing has no *cliffs*. From the exact oracle, share of decision states by how much the action
choice changes the outcome:

```
                       barely     MIDDLE     critical      C2 result
FrozenLake det         50.9%       0.0%       49.1%        CCE WINS
FrozenLake slippery    24.5%      62.3%       13.2%        null
Budget routing         10.9%      75.2%       13.9%        null
```

FrozenLake-deterministic — the only env CCE wins on — has an **empty middle**: every decision is
either free or critical. Routing is a smooth ramp, so "replay the top 10%" is an arbitrary cut
through a slope rather than a clump worth isolating.

Two properties of our design cause this, and both are ours to change:

1. **Catastrophe is impossible.** The return-feasibility mask means the truck can never strand
   itself. The worst available outcome is "served 6 instead of 7". There is no zero to fall to.
2. **Nothing is irreversible.** Skipping a customer now leaves it available later.

## What we change

**Time windows.** Each customer gets a window `[a_i, b_i]` in spend units. Arrive after `b_i` and
that customer is **gone for the rest of the episode**. This is the single most damaging constraint
in SVRPBench (NeurIPS 2025): +536-648% cost across all solvers, learning-based methods down to
85-88% feasibility while classical solvers held above 97%. It is nearly free for us because
`spent` already functions as a clock — no new state variable.

**Strandable shift.** Drop the return-feasibility rule from the action mask. If the vehicle cannot
get home by `B`, the episode **fails outright**. Reward becomes terminal: `+k` if you get home
having served k customers, `0` if you do not. That is FrozenLake's structure — sparse, all-or-
nothing — which is what makes per-seed outcomes bimodal instead of everyone landing on ~1.0.

Reinterpretation that makes both coherent: **`spent` is TIME, not fuel.** Driving advances the
clock, `B` is the end of the shift, windows are delivery appointments. Same state variable, and it
matches the standard VRPTW formulation rather than inventing one.

## The oracle survives, and that is the point

`V*(s)` = max customers servable on a route that **also gets home by B**, honouring windows.
Spend still strictly increases, so the state graph is still a DAG and one backward pass solves it.
States from which no successful return exists get `V* = 0` — which is exactly the cliff, and it is
computed exactly rather than estimated.

## Order of work — each gate can kill the plan cheaply

1. **Build** `time_windows=` and `allow_stranding=` flags on `envs/routing_budget.py`. Defaults
   OFF, so every committed budget-mode result still reproduces. Extend the oracle to match.
2. **GATE 1 — the stakes check (hours, no training).** Measure the middle-band mass with the
   exact oracle. Target: middle band well under 25%, ideally near FrozenLake's 0%.
   **If it does not move, STOP.** Do not build a trainer, do not spend a sweep. This is the whole
   reason we built the suitability tooling.
3. **GATE 2 — the learnability check.** Plain DQN-uniform must solve **30-60% of seeds** at the
   evaluation budget. Too high, still too easy; near zero, the sparse reward is unlearnable and we
   fall back to per-step reward with a stranding penalty. Calibrate window width and `B` here,
   using the BASELINE arm only.
4. **Pre-register.** Write E* (from the reference arm's solve-rate curve) and the predicted
   direction into the notebook BEFORE the sweep runs.
5. **Sweep.** 5 arms x 40 seeds, trained to E* rather than 4,000 episodes. Primary metric =
   **per-seed solve rate**, the FrozenLake metric, not AUC.

## What we are NOT doing, and why

- **Bigger instances** — tested. 12 customers vs 10, 360 runs, identical null. The tabular oracle
  also dies at 13 (4.4M states, 6.4 GB). Scale makes a problem harder to compute, not more pivotal.
- **More knob combinations** — that was the ~2,000 runs. Every cell null.
- **Adopting SVRPBench** — it evaluates constructive solvers with no replay buffer, so CCE has no
  attachment point, and it has no exact per-state ground truth. We take its constraint design and
  cite it; we do not run on it.

## Risks

- **Sparse terminal reward may be unlearnable.** Fallback: keep per-step `+1` and apply a large
  negative on stranding. Less bimodal, but learnable. Gate 2 decides.
- **State count grows** once the stranding mask is removed (more legal actions). Measure before
  committing to a sweep size.
- **Windows could make it trivially hard** — if too tight, nothing learns and we learn nothing.
  Gate 2 exists for this.
- Honest possibility: routing may simply lack pivotal decisions, and Gate 1 says so in hours. That
  is a publishable result about *when* CCE applies, not a failure.
