# Port the CCE suitability pipeline to Connect Four (FULL plan — all 6 metrics)

> This is the full-pipeline plan (the whole 4×3 grid, all metrics). The **probe** that gated it is
> a separate plan: `plans/cce-suitability-c4-probe.md`.

## PROBE OUTCOME (2026-06-20) — gate PASSED, design validated
Ran `probe_c4.py` (best player 259285, 93% WR, vs random foe, 600 boards, N=20). The predicted
degeneracy did **not** happen: `C_mean` median 0.22 (46% of boards > 0.25), `C_tv` median 0.37
(78.5% > 0.25, none near 0), and forks present at every game phase (early 0.21 / mid 0.21 / late
0.30). So the mcts-trained player vs a weak foe still faces real forks (blunders that hand the
random foe a win) → the weak-foe cells are informative, no filter/matched switch needed. Build the
full grid as planned.

## REUSE THE PROBE (it already solved the hard correctness bits)
`analysis/suitability/probe_c4.py` is the C4 backbone and must be factored into shared helpers
rather than reinvented. It already implements, correctly and tested: checkpoint loading
(`load_agent`), seat-normalized self-play board collection vs any foe (`collect_boards`), the foe
logic matching the rollout atom (`_foe_action`), masked greedy (`_greedy_action`), the batched
rollout-atom call with B-chunking + illegal→NaN (`score_boards`), and `C_mean`/`C_tv`
(`stakes` + `compute_consequence_metric`). The full build = lift these into `envs.py`/
`rollout_sweep.py`, add the remaining metrics (|TD|, NEED, concentration, SNR, horizon-fit), the
grid driver, and the scorecard plot.

## Context
We have a FrozenLake "will-CCE-help?" suitability pipeline (`analysis/suitability/`) that scores an
env with cheap rollout metrics (SNR, concentration, DISTINCT-TD, GAIN-FIDELITY, NEED, horizon-fit).
Next step: run the same idea on **Connect Four** to see whether the story (noise kills CCE; CCE ≠
PER; stakes concentrated/reachable) reads sensibly in a deep two-player env, and to produce the C4
**dose-response curve** (foe random → rule_based → mcts) that mirrors the FL slip sweep. Decision
tools for us, not paper figures.

Key realization: `metrics.py` is **env-agnostic** (plain numpy in) and reused unchanged except a
nan-safety tweak. The C4 consequence agent already exposes the rollout atom
(`_build_batched_rollout_fn` → `returns(B,K,N)`). So this is an adapter layer, not a rewrite.

## What's different in C4 (the real work)
1. **No oracle** → GAIN-FIDELITY = n/a (adapter returns `qstar_spread=None`; metric already yields None).
2. **States generated, not enumerated.** ~10^12 states, and a loaded checkpoint has an **empty
   replay buffer** (pickle stores params/target/opt/config/env_info — not the buffer). So we sample
   states by **rolling the loaded policy vs the foe** and collecting agent-to-move `pgx.State`s. Same
   rollout yields the visitation occupancy for NEED.
3. **Action masking.** Illegal (full) columns → set those `(s,a)` return slices to NaN; stakes
   `C(s)` = nanmax − nanmin over legal actions only.
4. **NEED = visitation occupancy** from the policy rollout (no tabular successor rep).
5. **Noise axis = the opponent** (random/rule_based/mcts) — the C4 analog of slip.

## The grid (decided)
One player family = **mcts-trained** run (259285), 4 checkpoints by win rate (weak/mid/strong/best),
each rolled vs {random, rule_based, mcts} = 12 cells. Read across a row = noise effect; down a
column = robustness across skill. **Scope "for now":** not matched, no benefit-validation yet →
results are descriptive, not predictive.

## Files to change
- **`envs.py`** — `make_connect_four_adapter(agent, name, opponent)`: collect agent-to-move pgx
  states + occupancy by self-play vs `opponent`; `qstar_spread=None`, `n_actions=7`, per-state
  `legal_mask`. **Apply foe pre-move seat-normalization (mirror `dqn.py:199-214`) so every state has
  `current_player==0`, then assert** — else the rollout reads the wrong player's reward (silent flip).
- **`rollout_sweep.py`** — C4 variants (branch on adapter family):
  - `compute_return_tensor` → `actions_array (B,7)` + `keys (B,7,N,2)`, call
    `agent._compiled_batched_fn(params, batched_states, actions, keys)`, illegal slices → NaN.
  - `greedy_actions`/`q_values` → `agent.network.apply` on `state.observation`, masked.
  - `compute_abs_td_per_state` → **no oracle**; sampled one-step bootstrap
    `|TD| = |r + γ·max Q_target(s') − Q(s,a_greedy)|`. Same greedy action as CCE. Estimate — label it.
  - `compute_occupancy_c4` → discounted visit counts from the rollout (replaces tabular version).
- **`metrics.py`** — nan-safety (all FL-safe): `stakes_C`→nanmax/nanmin; `snr`→nanmean/nanvar/
  nanmedian; `concentration`→nansum; `distinct_td`/`need`→mask finite rows before `spearmanr`.
- **`run_suitability.py`** — C4 path (sibling `run_suitability_c4.py`): build
  `Connect4ConsequenceDQN(ckpt['env_info'], config=ckpt['config'])`, load params, reuse
  `select_warmup_checkpoints`, loop checkpoints × foes. **Fresh agent per foe** (opponent is baked
  into `_compiled_batched_fn` at build time, NOT re-read on config change).
- **`scorecard.py`** — tolerate `gain_fidelity=None` (already does); add an opponent-sweep plot.

## Reuse (don't rewrite)
`metrics.py` (6 fns); `Connect4ConsequenceDQN`/`Connect4DQN.load`/`_build_batched_rollout_fn`/
`_compiled_batched_fn`/`_greedy_action`; `compute_consequence_metric`; opponents
`rule_based_action`/`mcts_action`/random step (`consequence_dqn.py:108-123`);
`select_warmup_checkpoints`/`_parse_metrics_log`.

## Cost & risks (from adversarial review)
- **MCTS foe is the cost wall** (~10^8 sims/cell at defaults). Cut `cf_n_rollouts` 30→~10,
  `mcts_n_sims` 32→~16, `cf_horizon` 42→~28, fewer boards/checkpoints; tiny-batch timing probe first.
- **Degeneracy on weak-foe cells** — strong player crushes weak foe → returns ~+1 → `C≈0` → flat
  SNR/concentration. Mitigations: probe first / contested-board filter / matched design.
- Conda env: `~/.conda/envs/counterfactual/bin/python` (not `bucks`).
- Parser/collection notes: prefer header-name parsing of metrics.log; extract collect+seat-normalize
  from `dqn.py:182-281` into a utility rather than stubbing `learn()`.

## Verification
1. Smoke: load a checkpoint, score ~8 boards (N=4) → `returns.shape==(B,7,N)`, illegal slices NaN,
   metrics finite, `gain_fidelity is None`.
2. FL regression: re-run FL pipeline after nan tweak → FL scorecard unchanged.
3. **PROBE gate** (separate plan) — `best × random` stakes spread healthy before scaling.
4. Full grid dose-response: SNR rises, stakes coarsen as foe strengthens; emit per-foe scorecards +
   combined SNR-vs-foe plot under `docs/figures/suitability/c4/`.

## Review round 2 (full-build) — fixes folded + staging
Metrics:
- **nan-safety in metrics.py** (5 fns, byte-identical for FL — confirmed): `stakes_C`→nanmax/nanmin;
  `snr`→nanmean/nanvar/nanmedian + `np.errstate(all='ignore')` guard for all-NaN slices;
  `concentration`→filter `C[isfinite]` before gini/sum; `distinct_td`/`need`→mask finite rows before
  `spearmanr`. Add a unit check (finite input identical; NaN input handled).
- **distinct_td**: write `compute_abs_td_c4` — one-step bootstrap `|r + γ(1-done)·max Q_target(s') −
  Q(s,a_greedy)|` via `pgx.step` (vmapped) + `agent.target_params`; **average a few foe replies**
  (random foe is noisy). Same greedy action as CCE.
- **NEED**: dedup destroys occupancy → `collect_c4_states` must track visit counts **before** dedup
  and return discounted occupancy aligned to the deduped state list (keep; cheap).
- **horizon_fit**: **DEFER v1** → return None (atom bakes `cf_horizon`; would recompile per H;
  scorecard already tolerates None). gain_fidelity: None (GO).
Integration:
- scorecard env keys = `C4-random/C4-rule_based/C4-mcts` (one line per foe in `plot_scorecard`,
  no rewrite); add `plot_opponent_sweep`; make `inject_dashboard` env_keys dynamic.
- **Fresh agent per foe** (opponent baked at build). Factor probe: `score_boards`→
  `compute_return_tensor_c4` (rollout_sweep.py); `collect_boards`→`collect_c4_states`+occupancy
  (envs.py); reuse `_greedy_action`/`_foe_action`; FL path untouched, no name clashes.
Cost & staging (measured: random ~4min/cell; rule_based ~12min; mcts ~14hr at defaults):
- Full grid at defaults = ~172 GPU-hr (infeasible) → mcts cuts `cf_n_rollouts` 30→10,
  `mcts_n_sims` 32→16, `cf_horizon` 42→28 → ~23 GPU-hr. chunk 64 safe (≤96 for mcts).
- STAGE: (1) C4 pipeline + nan-safety + smoke; (2) **random × 4 ckpts** (cheap, ~16min);
  (3) **rule_based × 4** (~50min); (4) **mcts timing probe** (best, 100 boards, cut budget) to
  measure real cost; (5) **mcts × 4** gated on (4). Build + cheap cells first; mcts last.

## Status
Probe gate PASSED. Review round 2 done, fixes folded. Building now (stage 1: nan-safety + C4
pipeline). Branch: research/cce-buffer-diagnosis.
