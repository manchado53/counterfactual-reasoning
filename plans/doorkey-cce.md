# CCE on MiniGrid DoorKey — 2nd good-oracle environment

**Branch:** research/andon-vending-bench-cce (this worktree) · **Status:** plan
**Why:** The lab notebook's "THE BAR" says C1 needs *one more good-oracle env* and C2 needs
*two more clean scenarios*. DoorKey is deterministic with a tiny enumerable state space, so it
can feed **both claims** — the strongest single thing we can add to the paper right now.

## The idea in one line
Reproduce the FrozenLake result (C1: CCE scores track an exact oracle; C2: CCE speeds learning)
in DoorKey, proving CCE works in a **second domain** with a genuinely different structure
(gating decisions instead of catastrophic holes).

## Why DoorKey fits the theory (from the paper)
- **Deterministic** transitions → no action-independent slip noise to dilute CCE (Corollary 1).
  FrozenLake-deterministic was the big win; slippery was the null. DoorKey is deterministic.
- **Sparse pivotal decisions** = the *gating* structure: you must pick up the key before you can
  open the door before you can reach the goal. Those two decisions gate the whole outcome
  (Theorem 1: big action-value gap → guaranteed big CCE). No holes/lava — consequence comes from
  discounting path length + gating, not catastrophe. Different mechanism, same thesis.
- **Enumerable → exact oracle.** State = (agent cell, direction, has_key, door_state) is a few
  hundred states, so value iteration on `env.P` gives exact Q* — the C1 ground truth we've only
  ever had on FrozenLake.

## Design decisions (defaults chosen — tell me to change any)
- **Reimplement in pure JAX** (`envs/doorkey.py`), mirroring `frozen_lake.py`: one construction
  pass emits both the Gymnasium-style `env.P` dict (for the oracle) and jit/vmap arrays (for
  fast rollouts + training). NOT pip-install minigrid/navix — neither exposes `env.P`, and this
  matches how FrozenLake was done. (Optional: install Farama `minigrid` only as a *reference* to
  cross-check our transitions; deferred, and /home is disk-full so kept optional.)
- **Fixed layout, DoorKey-6x6** (~200–300 enumerable states; comparable to FrozenLake-8x8's
  scale, richer than 5x5). Hard-code one split/door/key/agent layout so the state set is fixed
  and enumerable. 8x8 (~500–740 states) is a stretch option for a denser scatter.
- **State = single integer** indexing the enumerated (cell, dir, has_key, door_state). obs==state,
  so the Q-network (one-hot → MLP → ReLU), the CCE rollout, and the buffer are reused UNCHANGED.
- **Actions = full MiniGrid 7** (left, right, forward, pickup, drop, toggle, done). drop/done are
  harmless no-ops here → they just add low-consequence actions, which is fine for the CCE spread.
  Fallback lever if learning is slow: prune to 5 (drop drop+done).
- **Reward = +1 on reaching goal, 0 else, γ=0.99** (time-independent, so the tabular oracle is
  valid; γ<1 makes shorter paths + gating consequential). γ=0.95 is a lever to sharpen oracle gaps.
- **Score only REACHABLE non-terminal states** (BFS from start under all actions) for the C1
  correlation, so we compare on states the policy actually visits.

## What we build
1. `envs/doorkey.py` — JAX DoorKey: enumerate reachable states → `P[s][a]=[(1.0,s',r,done)]`
   + `next_states/rewards/dones` arrays; `reset(key)->(obs,state)`, `step(key,state,a)->
   (obs,next_state,reward,done,info)` (jit/vmap-pure), plus geometry metadata for the heatmap.
   Unit tests: transitions match hand-worked cases; optional cross-check vs Farama minigrid.
2. **Claim 1** (`analysis/claim1/doorkey/`): `oracle.py` (VI → mean action-gap label),
   `score_states.py` (7-action rollout scorer, total_variation, matches FL's max-aggregation),
   `heatmap.py` (DoorKey grid with key/door/goal), `run_analysis.py` (reuse Spearman +
   Precision@K + shared `scatter.py`). Needs 3 checkpoints/seed (untrained/mid/trained).
3. **Claim 2** (`agents/doorkey/`): near-mechanical copies of the FrozenLake trainers with the env
   swapped and `n_actions=7` (`config.py`, `dqn.py`, `dqn_vectorized.py`, `consequence_dqn.py`,
   `consequence_dqn_vectorized.py`, `train.py`, `run_experiments.py`). Buffers, CCE scoring,
   priority mixing, crossing-cadence — all reused as-is. Then small edits to
   `analysis/claim2/parse_logs.py` (DoorKey header + column schema + runs dir + eval_steps=upd*4)
   and a `doorkey` entry in `run_analysis.py`'s ENV_THRESHOLDS; reuse compute_metrics/plot_figures.

## Order of work (each gate must pass before the next)
1. Env + unit tests (correctness first).
2. Oracle: VI runs, produces sensible action-gap labels on the fixed layout.
3. **Sanity gate** (pre-flight checklist): plain vectorized DQN must SOLVE DoorKey (win≈1.0)
   before touching CCE. If plain DQN can't learn, nothing downstream is valid.
4. C1: train 3 seeds to get untrained/mid/trained ckpts → Spearman ρ(oracle, CCE) + Precision@K
   + scatter/heatmap. Success = ρ rising with training, like FrozenLake's 0.32→0.77→0.89.
5. C2: cluster sweep (5 algorithms × ~10 seeds) → IQM curves, final IQM, P(improvement).
   Success = CCE (esp. multiplicative) ahead of DQN+PER on sample efficiency.
6. Log results to lab-notebook, update the paper's coverage table.

## Risks / watch-items
- **Weaker oracle spread than FrozenLake** (no catastrophic holes; consequence is gating +
  discount). Mitigation: lower γ to sharpen; if still flat, that itself is an informative result.
- **Reachability**: enumerate the true reachable set (BFS) so oracle & CCE align and dead states
  don't pollute the correlation.
- **cf_horizon must cover a full episode** (pre-flight checklist).
- **parse_logs is format-coupled**: emit a `metrics.log` header the parser attributes to DoorKey,
  or it silently parses SMAX columns.
- **Editable install points at MAIN repo** (memory): jobs launched from this worktree load
  main-repo code unless we prepend the worktree `src` to PYTHONPATH. Handle in the launch scripts.
- Compute in the `counterfactual` conda env, on Rosie SLURM; dry-run every sweep first.

## Execution note (ultracode on)
Once the env passes its tests, the trainer copies and the Claim-1 files are independent and can be
generated + adversarially verified in parallel via a Workflow (skeptics check the transition table
and the oracle against a reference). The env itself is the sequential bottleneck and is built first.
