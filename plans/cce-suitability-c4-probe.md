# C4 suitability PROBE — does the design even have stakes? (gate before the full grid)

> Sibling of the full plan `plans/cce-suitability-connect-four.md`. This is the small, first thing
> we build. It answers ONE question with data, then we decide whether to build the rest.

## The one question
When the **best mcts-trained player** plays the **random foe**, do the boards it actually faces have
real **stakes** (forks where the move decides the game), or are they degenerate (`C(board) ≈ 0`
because the strong player wins no matter what)?

```
healthy spread of C(board)  -> design is fine -> build the full 4x3 grid
nearly all C(board) ≈ 0      -> design hollow on weak foes -> switch (filter / matched)
```

## Why a probe (not the whole pipeline)
The degeneracy risk is a reviewer *prediction*, not measured. Random foe is cheap (~600–1000× less
than mcts), so we can test the riskiest assumption for minutes of compute before touching the shared
`metrics.py` / `rollout_sweep.py` / `envs.py`. Keep blast radius tiny.

## Design: ONE standalone script, no shared-file edits yet
New file: **`src/counterfactual_rl/analysis/suitability/probe_c4.py`** (self-contained). Reuses the
agent's own rollout atom; does NOT modify metrics.py/rollout_sweep.py/envs.py (that's the full plan,
only if the gate passes).

### Steps the script does
1. **Load the best player.** Run dir `agents/shared/runs/259285`. Build
   `Connect4ConsequenceDQN(ckpt['env_info'], config={**ckpt['config'], 'opponent':'random', ...cf
   overrides})`, then assign params directly (`agent.params/target_params = jax.tree.map(jnp.array,
   ckpt['params'/'target_params'])`) — construct-with-config THEN assign params, so the opponent is
   'random' before the rollout fn is built (do NOT rely on `agent.load()` if it overwrites config).
   Use `best.pkl` (run dir already has it) or the highest-win checkpoint.
2. **Collect boards by self-play vs random (seat-normalized).** Use `pgx.make('connect_four')`.
   `state = env.init(key)`; if `current_player != 0`, apply ONE random foe pre-move (mirror
   `dqn.py:199-214`) so it's our turn. Then loop: record the current `pgx.State` (assert
   `current_player==0`); agent greedy masked action → `env.step`; random foe masked reply →
   `env.step`; record next our-turn state; until terminal. Collect ~300 agent-to-move boards across
   ~enough games. Random foe step must match the atom's random step (`consequence_dqn.py:121`:
   masked argmax of `random.normal`).
3. **Score stakes C(board).** `agent._build_batched_rollout_fn()` (opponent='random' baked in).
   Stack collected states → `batched_states` (leaves `(B,...)`). `actions_array=(B,7)` = all columns;
   `keys_array=(B,7,N,2)`. `returns = agent._compiled_batched_fn(params, batched_states, actions,
   keys)` → `(B,7,N)`. Per board: `m = returns.mean(axis=2)` `(B,7)`; set illegal columns
   (`~legal_action_mask`) to NaN; `C(board) = nanmax(m,1) − nanmin(m,1)`.
4. **Report the distribution.** Print mean / median / p90 of `C(board)`, fraction with `C>0.25`,
   fraction `C<0.05`, and the mean game length collected. Save a histogram to
   `docs/figures/suitability/c4/probe_best_random_Cdist.png`. Print 3 example boards with their
   per-column mean returns so we can eyeball a real fork vs a flat board.

## Knobs (probe-scale)
`N rollouts = 20`, `cf_horizon = 42` (full game; cheap vs random), `~300` boards, atom batch chunked
(e.g. B-chunks of 64) to bound GPU memory. All overridable by CLI flags.

## Run
```
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.analysis.suitability.probe_c4 \
    --run-dir src/counterfactual_rl/agents/connect_four/.../runs/259285 \
    --n-boards 300 --n-rollouts 20 --out docs/figures/suitability/c4/probe_best_random.json
```
(Confirm the real path of run 259285 — it's under `agents/shared/runs/259285`.)

## Risks to confirm in review (before coding)
- Does `agent.load()` overwrite the config we set (would re-bake the wrong opponent)? → use
  construct-then-assign-params instead.
- Do collected states carry a usable `legal_action_mask`, and is `state.observation` the right
  `(84,)`/perspective input for greedy? (mirror `dqn.py:_greedy_action`).
- Does the atom assume `current_player==0`? (yes — our seat-normalized collection must satisfy it).
- Memory: `B×7×N` rollouts vmapped over horizon 42 — chunk B if OOM.
- Is `best.pkl` the right "best" (vs late-collapse)? cross-check with logged win rate.

## Gate / decision
- **Healthy** (e.g. median C ≳ 0.2, a clear right tail, >~30% boards with C>0.25): design holds →
  proceed to the full plan, next tackle rule_based + mcts cost.
- **Degenerate** (most C≈0): switch — add a contested-board filter (drop near-terminal blowouts) or
  go matched (random-trained player vs random foe), then re-probe.

## Review fixes (folded in — must implement)
1. **Random foe consistency.** The atom's random foe = `argmax(masked random.normal)`
   (`consequence_dqn.py:120-123`); the collection seat-norm/foe steps must use the SAME logic, NOT
   `jax.random.choice` uniform — so the foe that generates boards == the foe that scores them.
2. **Gate on BOTH stakes.** `C_mean` = nanmax−nanmin of per-column mean returns AND `C_tv` = the
   total-variation consequence (reuse `compute_consequence_metric`, the metric CCE actually uses).
   A board can have equal means but different spread (TV fork the mean misses). Healthy gate = both
   `C_mean ≳ 0.2` AND `C_tv ≳ 0.3`.
3. **De-bias the board sample.** Dedup boards (by state key); stratify ~early/mid/late game phase;
   bump to ~600 boards; report effective sample size (lag-1 autocorr) since within-game boards are
   correlated and short blowout games over-sample easy early positions (false-degenerate risk).
4. **Illegal forced move.** Confirm `pgx.step(state, illegal_col)` is safe (quick test); mask
   illegal columns to NaN BEFORE max−min AND exclude them from the TV alt-set — else an illegal-move
   loss (~−1) inflates C and fakes a fork.
5. **Loading.** `agent.load()` does NOT overwrite `self.config` — so construct with
   `config={'opponent':'random', ...}` then `agent.load(best.pkl)` (restores params, rebuilds jit).

## Status
Probe stage, review done, fixes folded. Branch: research/cce-buffer-diagnosis. Full plan parked in
`plans/cce-suitability-connect-four.md`.
