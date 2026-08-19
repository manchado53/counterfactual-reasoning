# CCE on Vehicle Routing (CVRP) — a logistics env for Claim 1 + Claim 2

**Branch:** research/cce-cvrp-logistics (new, off master) · **Status:** plan (awaiting sign-off)
**Why:** The paper's C2 "speeds learning" win is clean only on deterministic FrozenLake; C1's
exact oracle has only ever existed on FrozenLake. We need (a) a *second* clean C2 domain and
(b) a *second* exact-oracle C1 domain — ideally real-world, not another toy grid. Vehicle routing
is logistics (a different domain), it is **deterministic** (CCE's proven sweet spot — the
graded-slip knife-edge says CCE wins at determinism), its decisions are **sparse + pivotal**
(a few "which way to branch" choices decide the whole route), and small instances have an
**exact dynamic-programming oracle**. Same thesis as FrozenLake/DoorKey, genuinely different
structure (routing cost, not holes or gating).

## The idea in one line
Reproduce the FrozenLake/DoorKey result — C1: CCE scores track an exact oracle; C2: CCE speeds
DQN learning — in capacitated vehicle routing, proving CCE works in a real logistics domain.

## BUILD STATUS (2026-08-06)
- **DECIDED: ported our own** (below). `envs/cvrp.py` — TSP + CVRP modes, 14/14 tests pass.
- **Oracle done + validated**: `analysis/claim1/cvrp/oracle.py`, exact backward induction,
  matches BRUTE FORCE on both TSP (all permutations) and CVRP (permutations x optimal
  load-split). 7/7 tests. Carries a Bellman-residual self-check — it caught a real bug where
  the CVRP reload edge (customer -> depot, same visited-mask) broke the popcount topological
  order; fixed with a depot-first secondary sort key.
- **Trainer done**: `agents/cvrp/{config,dqn,sanity}.py`, masked node-selection DQN.
- **SANITY GATE PASSED** on the small instance: plain DQN-uniform reached **1.0000 of optimal**
  (random policy = 0.77). Full 10-customer instance in progress.
- Metric: `opt_ratio` = optimal_length / achieved_length in (0,1], 1.0 = provably optimal.
  Bounded and higher-is-better, so the rliable/IQM machinery carries over unchanged.

## FINDING THAT SHAPES CLAIM 2 — routing has LOW stakes-concentration
Measured with the same Eq-7 oracle and the same code on both envs:
```
                      gini   max/median   "dead" states (<5% of max stakes)
FrozenLake 8x8 det    0.559     35.9x            50.9%
FrozenLake 8x8 slip   0.546     10.2x            30.2%
routing TSP           0.208      1.9x             0.7%
routing CVRP cap10    0.245      2.7x             3.0%
routing CVRP cap6     0.314      2.9x             9.8%
routing CVRP cap5     0.361      3.2x            15.3%
```
In a grid you can stand on open ice where all moves are equivalent; in routing EVERY choice
picks among spatially spread stops, so stakes are near-uniform. Confirmed across five very
different geometries (ring / clustered / hub+outliers / hub+1remote / two-lobes): gini stayed
0.19-0.27 regardless. It is a property of the domain, not of the layout.

**The capacity limit is what creates pivotal moments**, exactly as predicted — and the effect
is large and monotone. Splitting decisions by where they are made:
```
                  at-a-stop stakes   at-depot stakes    ratio
CVRP capacity 10       0.330              0.057          5.8x
CVRP capacity 6        0.306              0.015         20.6x
CVRP capacity 5        0.293              0.007         42.2x
```
Caveat: at-a-stop states vastly outnumber at-depot ones, so routing still lacks FrozenLake's
*sparse* pivotal set — the rocks are real but they are everywhere.

**Consequences for the claims**
- **Claim 1: GO.** Spearman needs only a monotone ranking, and the oracle spreads fine.
- **Claim 2: predicted WEAK**, and this prediction comes from the June suitability pipeline
  (`docs/SUITABILITY_METRICS.md`) — this is that tool earning its keep before we spend compute.

**NEW EXPERIMENT — the CAPACITY DIAL (approved).** Same dose-response design as the slip
sweep, on a new axis: sweep capacity in {10, 6, 5} x 5 arms x ~10 seeds and test whether CCE's
benefit TRACKS stakes-concentration. Tighter capacity -> more concentration -> (predicted)
larger CCE advantage. A flat result falsifies the suitability thesis cheaply; a tracking result
validates it on an axis independent of noise. `capacity` is already a config knob.

## KEY DECISION (RESOLVED — ported our own) — port our own vs install Jumanji
Preflight: **neither jumanji nor ortools is installed**, and jax is **0.9.1** (very recent).
- **Recommended: port a tiny CVRP ourselves in pure JAX** (`envs/cvrp.py`), mirroring
  `frozen_lake.py` / `doorkey.py`. Reasons: (1) installing jumanji may downgrade/clash with
  jax 0.9.1 and **break the shared `counterfactual` env** other live experiments depend on;
  (2) we write the oracle ourselves either way, so Jumanji only saves the (simple) env dynamics;
  (3) it matches the precedent the team already chose twice (FrozenLake and DoorKey were both
  reimplemented, not pip-installed, precisely to expose the transition structure for the oracle);
  (4) **zero new dependencies** — the routing dynamics are ~150 lines and a Held-Karp oracle is
  ~30 lines of numpy. CVRP dynamics are simpler than DoorKey's grid geometry.
- **Fallback: install jumanji + ortools in a throwaway env, use its CVRP.** Only if we decide the
  off-the-shelf dynamics are worth the dependency risk. Same env either way from CCE's view.

## Design defaults (tell me to change any)
- **Fixed small instance** — ~10 customers + 1 depot, fixed coordinates (like DoorKey's fixed
  layout). One MDP the DQN can master and the DP can label exactly. (Random maps each episode →
  a harder generalization problem → v2.)
- **Reward = dense, negative step distance** (good DQN signal), sparse tour-length as a lever.
- **Capacity + depot reloads** = the "capacitated" part (real logistics). **TSP-first** (no
  capacity) as first-light to prove the pipeline before adding capacity.
- **State = (current node, visited mask, remaining capacity)**; obs = one-hot current + visited
  bits + capacity scalar → reuse the MLP DQN unchanged except the action head.
- **Actions = pick next node (N+1)** with **action masking** (invalid/served/over-capacity → set
  Q = −inf before argmax and before the CCE per-action spread).
- **γ = 0.99**, cf_horizon = full episode (= N steps).

## Oracle (C1 ground truth)
Exact DP over the fixed instance: optimal completion cost from any (current, visited-set,
capacity) → per-decision **regret** of a next-node choice = ground-truth importance (analogue of
FrozenLake's value-iteration Q*-spread). TSP uses plain Held-Karp (O(2^N·N²), fine at N≈10). No
solver dependency needed. Score only reachable states.

## What we build (mirror DoorKey / frozen_lake)
1. `envs/cvrp.py` — JAX routing env: `reset(key)->(obs,state)`, `step(key,state,a)->
   (obs,next_state,reward,done,info)` (jit/vmap-pure) + structure for the oracle + geometry
   metadata for the route picture. Unit tests: distances, masking, capacity, termination.
2. **Claim 1** (`analysis/claim1/cvrp/`): `oracle.py` (DP → per-decision regret), `score_states.py`
   (N-action rollout scorer, total_variation, max-aggregation like FL), `plot.py` (route with
   per-decision importance), `run_analysis.py` (reuse shared `scatter.py` + Spearman + Precision@K).
   Needs 3 checkpoints/seed (untrained/mid/trained).
3. **Claim 2** (`agents/cvrp/`): near-mechanical copies of the FrozenLake trainers with the env
   swapped and an N+1 masked action head (`config.py`, `dqn.py`, `dqn_vectorized.py`,
   `consequence_dqn.py`, `consequence_dqn_vectorized.py`, `train.py`, `run_experiments.py`).
   Buffers, CCE scoring, priority mixing, crossing-cadence reused as-is. Then a `cvrp` entry in
   `analysis/claim2/parse_logs.py` + `run_analysis.py` ENV_THRESHOLDS.

## Order of work (each gate must pass before the next)
1. Env + unit tests (correctness first).
2. Oracle: DP runs, produces sensible optimal-completion / regret labels on the fixed instance.
3. **Sanity gate** (pre-flight checklist): plain vectorized DQN must SOLVE the instance
   (near-optimal tour) before touching CCE. If plain DQN can't learn it, nothing downstream is valid.
4. C1: 3 seeds → untrained/mid/trained ckpts → Spearman ρ(oracle, CCE) + Precision@K + route plot.
   Success = ρ rising with training, like FrozenLake's 0.32→0.77→0.89.
5. C2: cluster sweep (5 algorithms × ~10 seeds) → IQM curves, final IQM, P(improvement).
   Success = CCE (esp. multiplicative) ahead of DQN+PER on sample efficiency.
6. Log to lab-notebook; update the paper's coverage table; update the env report.

## Risks / watch-items
- **DQN on routing is unusual** (SOTA uses attention/pointer nets). Mitigation: fixed small
  instance = a single deterministic shortest-tour MDP a plain MLP DQN can master; we test a
  *replay* method, not best-in-world routing. If plain DQN can't solve N=10, drop to N=8 / TSP.
- **Action masking** must apply to the Q-values everywhere (training argmax, eval, and the CCE
  counterfactual action set) — set masked Q/scores to −inf, don't let the agent "teleport."
- **cf_horizon = full episode**; **consequence_metric = total_variation** (pre-flight checklist).
- **CCE per-action spread cost scales with N** (N counterfactual rollouts/decision) → keep N small.
- **Determinism** means the noise / Corollary-1 test does NOT live here — that's a later
  stochastic-demand variant. Here we get the clean determinism C2 win + the new-domain C1 oracle.
- **Editable install points at MAIN repo** (memory): jobs from the worktree load main-repo code
  unless we prepend the worktree `src` to PYTHONPATH. Handle in launch scripts; never `pip -e` the worktree.
- Compute in the `counterfactual` conda env, on Rosie SLURM; dry-run every sweep first.
