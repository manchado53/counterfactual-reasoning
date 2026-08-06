# DoorKey + Lava — giving CCE a real cliff to find

## Context
The no-lava DoorKey-6x6 experiment (branch `experiment/cce-doorkey`, pushed) gave a real but
capped Claim-1 result (rho 0.13→0.45→0.43) and a clean Claim-2 null (P(CCE>PER)=0.550
uniformly, PER fastest). Diagnosed cause for both: DoorKey is fully reversible — no wrong
action ever ends the episode, it just costs a few extra discounted steps. That means the
oracle's own action-value gaps are small almost everywhere (checked: only ~2x spread across
all 152 states), so there is no sharp signal for CCE's total-variation score to lock onto, and
no sample-efficiency edge for it to win on. FrozenLake's holes are exactly this kind of cliff —
step wrong, game over, Q drops from ~1 to 0 instantly — and that cliff is what both of
FrozenLake's C1 (rho 0.89) and C2 (mul IQM 1.00 vs PER 0.46) results actually run on.

Lava tiles are the direct, MiniGrid-native way to add that cliff to DoorKey while keeping its
own distinguishing structure (the key/door gate) intact — so the paper gets a domain that
combines catastrophe *and* gating, which is novel relative to plain FrozenLake.

Decisions locked with the user: **new branch `experiment/cce-doorkey-lava`** (keeps the clean
no-lava null citable on its own branch), **move to DoorKey-8x8** (13→31 free cells — enough
room for lava to create a real "safe detour vs risky shortcut" choice rather than just
blocking the only path).

## What lava does, mechanically (MiniGrid-faithful)
Lava is **walkable but fatal**: `FORWARD` onto a lava cell succeeds (like walking onto the
goal), but ends the episode with reward 0 (like FrozenLake's holes) — not a wall. This is
exactly MiniGrid's own `Lava.can_overlap()=True` + terminate-on-entry semantics.

```
near lava:  action A (detour)      -> eventually reach goal, return ~ high
            action B (step in lava) -> DEAD, return = 0
            Q-gap: LARGE. Oracle spikes here. CCE's rollout TV spikes here too.
elsewhere:  same as before (small gaps) — unaffected, still gated by key/door.
```

## Layout (DoorKey-8x8 + 4 lava tiles)
Same split-wall-with-one-door pattern as the 6x6, scaled up: interior rows/cols 1-6, split
wall at col 4 (door at row 3), left room cols 1-3 (start (1,1), key), right room cols 5-6
(goal at (6,6)). Two lava tiles in the left room (between start and key) and two in the right
room (between door and goal), so **both legs of the trip** carry real risk, not just the gate
itself. A verified-by-hand safe route exists around all four (confirmed by hand-tracing during
design; will be re-confirmed programmatically by the BFS solvability test). Exact coordinates
are an implementation detail I'll finalize against the solvability + reachability checks, not
something to lock in the plan.

## What changes in code

### 1. `envs/doorkey.py` (env owner)
- Add `LAVA = "L"` tile glyph.
- `_find_tile` only returns a single match (used for key/door/goal, each unique) — add
  `_find_all_tiles(glyph)` returning a list, used for `self.lava_cells`.
- Generalize the existing goal-only absorbing check into `_is_terminal(row,col)` = goal OR
  lava; keep the `+1` reward exclusive to the goal-entering transition (lava-entering
  transition = reward 0, done=True, mirrors how FrozenLake's hole entry already works).
- Add the new `DOORKEY_8x8` layout (with lava) to `LAYOUTS`; keep `DOORKEY_6x6` as-is (still
  used for reference/tests, not deleted).
- Update `agents/doorkey/train.py`'s `--layout` choices to include `'8x8'`.

### 2. `tests/test_doorkey_env.py`
Add lava-specific hand-worked transitions (forward onto lava = terminal, reward 0; lava state
absorbs like the goal does) and extend the BFS solvability test to confirm a safe zero-lava
path exists on the 8x8 layout.

### 3. Everything else — reused as-is
`analysis/claim1/doorkey/{oracle,score_states,heatmap,run_analysis}.py` and
`analysis/claim2/parse_logs.py`/`run_analysis.py` already take `layout_name`/`slip_prob` as
parameters and are layout-agnostic (grid dims come from `env.nrows/ncols`, geometry from
`env.desc`) — no changes needed beyond passing `layout_name='8x8'`. `heatmap.py` already
renders any glyph it doesn't recognize as plain floor; I'll add lava (`L`) as a distinctly
marked tile (small addition, mirrors the existing key/door/goal glyph handling) so the figure
actually shows where the cliffs are.

## Order of work (gates — same discipline as last time)
1. **Env + tests** — lava transitions correct, BFS-confirmed safe route exists, reachable
   count sane (~500-700 states, still trivial for VI).
2. **Independent adversarial audit** (same pattern as before — a from-scratch reference
   re-derives the transition table and oracle; caught nothing last time but is cheap and has
   real teeth via mutation testing).
3. **Cheap pre-training preview** (like the slip discovery): using the *optimal* Q* policy,
   check CCE-vs-oracle rho at both slip=0 and slip=0.2 *before spending any GPU*. This also
   answers a real open question: does lava's cliff give even the **deterministic** rollout
   enough spike to work for C1 (spike-vs-spike TV can be meaningfully 0 or 1, unlike the flat
   no-lava case), or do we still need slip>0? Cheapest possible validation gate.
4. **Sanity gate** — plain DQN still solves DoorKey-8x8+lava reliably (watch for: lava
   punishing early high-epsilon exploration so hard that episodes rarely complete — if so,
   may need a softer epsilon schedule or more episodes, not a design change).
5. **C1 sweep** — 3 seeds, whichever slip level step 3 validates, checkpointed
   untrained/mid/trained → Claim-1 analysis. Target: rho meaningfully higher than the no-lava
   0.43 ceiling, ideally approaching FrozenLake's range.
6. **C2 sweep** — 5 algorithms × 10 seeds, deterministic (slip=0) → Claim-2 analysis. Target:
   CCE (esp. multiplicative) pulls ahead of PER now that near-lava transitions are genuinely
   high-value to replay — the same mechanism behind FrozenLake-deterministic's win.
7. **Log + push** — dated lab-notebook entry, commit code+figures, push
   `experiment/cce-doorkey-lava`.

## Verification
Same as the no-lava round, which caught a real bug last time — repeat the discipline:
- Env: unit tests + independent adversarial audit before any training.
- Pre-training rho preview before spending GPU (this alone saved a wasted C1 sweep last time
  by predicting the slip requirement in advance).
- Sanity gate must pass before the sweeps launch.
- Explicitly re-verify the CCE scoring env config matches the training slip_prob (the exact
  bug found last round) — `score_states.py` already takes `slip_prob` as an explicit required
  parameter now, so this class of bug is structurally harder to reintroduce, but I'll double
  check the call site again before trusting any C1 numbers.
- C2: check every run's `sacct` state, not just win-rate — last round two "0% win-rate" runs
  were actually cluster SIGKILLs, not real failures, and the fix mattered for the final numbers.

## Gotchas carried over (still apply)
- Worktree PYTHONPATH: jobs must prepend `<worktree>/src`, never rely on the editable install
  (it points at the main repo).
- Vectorized crossing-logic cadence (`total//f > prev//f`), target-freq conversion.
- `cf_horizon` must cover a full episode (8x8's optimal path is longer than 6x6's 11 steps —
  will re-measure and set accordingly, likely 60-100).
- `/home` disk near-full — keep large run outputs out of git (already gitignored via
  `**/runs`, `**/experiments/`), only commit code + figures.
