# CCE on budget-constrained routing — fixing the CVRP Claim-2 null

**Branch:** research/cce-cvrp-logistics · **Status:** built, 600-run sweep launched 2026-08-18
**Sweep:** SLURM job 273322 (array 0-59) · manifest `experiments/2026-08/budget_dial.jsonl`

## Why
CVRP gave us Claim 1 (rho 0.52 -> 0.67, exact oracle, 3 seeds) but Claim 2 was a flat null
over 50 runs — no arm beat PER, and uniform random replay "beat" PER at traffic-off, which
is the signature of a noise-dominated measurement. Two causes were diagnosed, and both are
properties of the REWARD rather than of routing:

1. **No headroom.** Reward = -distance is dense and easy; plain DQN hit 1.0000 of optimal by
   episode ~750 of 14,000. Replay order cannot matter after the task is already solved.
2. **Saturated score.** CCE scores a state by the total variation between its actions' return
   distributions. Under a continuous deterministic reward every action returns a different
   real number, so TV = 1 at ~100% of states. A score that fires everywhere ranks nothing.

The rule that came out of it: **the TV score needs outcomes that can TIE** — a discrete
reward — or stochasticity. FrozenLake's 0/1 reward ties constantly, fires on ~4% of states,
and CCE wins there.

## The change
Switch the goal from "drive short" to the **Orienteering Problem**: serve as many customers
as possible on a closed tour within a travel budget B.

- reward becomes an **integer count** (0..N) -> outcomes tie -> TV grades again
- **B sets the difficulty directly** -> headroom is a knob, not a hope
- prior art is deep (distance-constrained VRP, Laporte/Desrochers/Nobert 1984; the
  Orienteering Problem and its Team/Time-Window variants), so this is a recognized OR
  variant, not an env invented to make our method win
- the oracle stays **exact**: budget-spent strictly increases, so the state graph is a DAG
  and one backward pass gives max-servable from every state

Budget-spent must be in the state or the MDP is not Markov, so distances are quantized to
integer units (`dist_scale=10`). That is not an approximation — the instance is *defined* on
those integers and the oracle solves that instance exactly.

## What was measured BEFORE running (this is the pre-registration)
Exact-oracle stakes concentration and a plain-DQN headroom gate:

```
budget_mult   B(u)   states    optimal served   gini   dead states   DQN-uniform
   0.55        19     4,707        5/10         0.222     12.2%      -
   0.75        26    47,582        8/10         0.214     11.4%      0.750 -> 0.875, climbing
   0.95        33   183,826        9/10         0.262     16.3%      0.889 flat from ep 400
   1.30        46   382,195       10/10         0.369     30.3%      1.000 at ep 400 (CEILING)
FrozenLake 8x8 det (CCE's win)                  0.559     50.9%
```

**My first prediction was wrong and the measurement corrected it.** I expected a tight budget
to concentrate the stakes; it does the opposite — when everything is on a knife edge, ~88% of
states have stakes and the signal is flat again. Loose budgets concentrate the stakes but
remove the headroom.

**Registered prediction: an INVERTED U.** CCE's advantage over PER should peak in the MIDDLE
of the dial. Falsifiable three ways: flat, monotone up, or monotone down all contradict it.

## The sweep
600 runs = 5 budgets {0.60, 0.70, 0.80, 0.90, 1.00} x 2 capacities {10, 6} x 5 arms
{uniform, PER, CCE-only, CCE+TD add, CCE+TD mul} x 12 seeds. 4000 episodes, eval every 25
(the learning all happens before ~ep 1500, so eval resolution matters more than length).
Capacity is a second, independent dial — a real effect should show on both axes.

Metric: `opt_ratio` = served / max-servable in [0,1], so the existing rliable/IQM machinery
carries over. Primary comparison = AUC of the eval curve; P(beats PER) by stratified bootstrap.

## Watch-items
- 5 seeds was too few last time (uniform "beat" PER). 12 seeds here, and P(>PER) is reported
  with the bootstrap rather than eyeballed means.
- `opt_ratio` is COARSE in budget mode (served is an integer, so the curve steps by 1/optimal).
  AUC over a 160-point curve is the mitigation; if arms still cannot be separated, that is a
  measurement limit to report, not a null to claim.
- Jobs run from a git worktree: the editable install points at the MAIN repo, so the sbatch
  script puts the worktree `src` first on PYTHONPATH.

## Files
- `envs/routing_budget.py` — the env (+ `optimal_closed_tour_units` for the denominator)
- `analysis/claim1/cvrp/budget_oracle.py` — exact DP, Bellman self-check, brute-force reference
- `tests/test_routing_budget.py` — 18 tests; oracle == brute force at six budget settings
- `agents/cvrp/budget_experiments.py` — the sweep grid
- `analysis/claim2/cvrp_budget_sweep.py` — parser, bootstrap, dial figure
