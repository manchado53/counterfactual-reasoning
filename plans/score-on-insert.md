# Score-on-insert — 100% coverage, plus the reward arm

## Why

CCE's replay signal is ~84% fabricated. A slot is scored only if a random draw happens to
hit it, and over its whole 6,250-update life it gets 0.32 expected draws — so 73% of slots
die carrying the running mean they inherited on insertion, and the sampler cannot tell those
apart from real measurements.

Raising coverage by drawing more often is the coupon-collector trap: 90% coverage costs
1,000,000 rollouts per buffer turnover (31x today). Visiting each arrival exactly once costs
100,000 (3.1x) and reaches 100%.

**The second half of the experiment is the reward.** With `coll_rew=0.0` and
`sparse_reward=True`, a crash and a timeout both return 0 — they are the same number. So
next to a wall, where every action crashes, the divergence among per-action returns is
exactly 0. That is CCE correctly reporting "the action choice changed nothing", and it
explains three separate findings: 47.8% of map positions scoring exactly 0, CCE correlating
**+0.688** with distance-to-obstacle, and 28.5% zeros even with bootstrap on.
`coll_rew=-1` makes crashing genuinely worse than wandering, and consequence should then
appear exactly where the obstacles are.

## The design (user's call: both horizons, both rewards)

2 x 2, all `weighted_mean` + `rank`, 100% coverage, 5 seeds.

```
                 coll_rew = 0.0          coll_rew = -1.0
   H = 60        soi_h60_r0              soi_h60_rneg
   H = 200       soi_h200_r0             soi_h200_rneg
```

Plus baselines for the NEW reward cell only (the `coll_rew=0` cell already has 26 uniform
and 18 PER runs): `uniform` and `per` at `coll_rew=-1`, 5 seeds each, ~14 GPU-h.

**Recorded disagreement, so it is on the record and not re-argued:** H=200 is measured at
~89% zero scores against H=60's 28.5%, because `max_steps=200` means the episode ends before
the horizon and `cf_bootstrap` never fires. I expect the H=200 half to be 100% coverage of
mostly-zeros. It runs anyway — it is the cell nobody has ever measured at full coverage, and
the smoke will settle it for a few GPU-hours instead of an argument.

## Settled knobs — do not re-open

```
   buffer_capacity      100,000   everything below 50k collapses (array 275562);
                                  and capacity buys no coverage above the burst size
   cf_rollout_temp          0.0   greedy -> R=1 is EXACT -> 20x cheaper
   cf_n_rollouts              1   same reason
   cf_bootstrap            True   without it 82-89% of scores are exactly 0
   consequence_metric  wasserstein  TV returns exactly 0.0 at len<2, so R=1 forces it
   priority_mixing   multiplicative, mu_c=1, mu_delta=1
   aggregation        weighted_mean   (max is significantly WORSE than uniform, p=0.048)
   normalise_priorities    rank   only mode with a principled basis (cv 47.4 -> 0.58);
                                  costs +6.5h flat per run
   fill 0.3 / 8x8 / 250k episodes / 5 seeds
```

## Build

**1. `agents/shared/consequence_buffers.py`**
- `add_batch()` returns the array of indices it wrote. Today it returns `None` and loops
  `add()` (line 185-197); the write positions are `(write_pos_before + i) % capacity`.
  Additive, so FrozenLake / SMAX / Connect Four callers are unaffected.

**2. `agents/jax_nav/consequence_dqn.py`**
- Split `_score_buffer_transitions()` (line 184). It already works from an index list — only
  the *source* changes. New `_score_indices(indices)` holds the body; the old method becomes
  `self._score_indices(self.buffer.sample_uniform(n_score)[1])`.
- **Chunking is mandatory.** 8,192 transitions x 15 actions x 1 rollout = **122,880 parallel
  envs**, 6.4x the widest batch ever run here (greedy_cov's 19,200, which is the proven-safe
  T4 width). `_score_indices` must chunk to `cf_score_chunk` (default 1,280 -> 19,200 envs)
  and loop. Without this the run OOMs on the first burst.
- New config knob `score_on_insert: bool = False`. Default off means every existing run
  reproduces bit-for-bit.
- When on, the `score_interval` gate at line 390/402 is skipped — scoring is driven by
  insertion, not by the update counter.

**3. `agents/jax_nav/consequence_dqn_vectorized.py`**
- After `self.buffer.add_batch(...)` (line 96), if `score_on_insert`, call
  `_score_indices()` on the returned indices.

**4. `slurm/sweep_score_on_insert.py`** — copy `sweep_capacity_floor.py` (array submission,
base64 config, manifest, `--dry-run` all present). Guard: raise `SystemExit` if
`score_on_insert` is set together with a non-null `score_interval`.

**5. `analysis/claim2/greedy_learning_curves.py`** — a `_label_soi` labeller keyed by
`(horizon, coll_rew)`. **The `coll_rew=-1` runs MUST NOT merge into the existing uniform/PER
baselines** — that is the same contamination shape as the `map_size` bug and the capacity
bug. Add `coll_rew` to the baseline guard beside `_is_default_capacity`.

## Cost, and the smoke gate

```
   per seed          H=60   34 - 51 h        (27-44h scoring + 6.5h rank)
                    H=200   58 - 152 h

   20 CCE runs + 10 baselines
   -> 934 - 2,044 GPU-h,  longest single run 2.4 - 6.3 days
   -> teaching TIMELIMIT is 7 days, so it fits, but only just at the top
```

The spread is wide because scoring cost is sublinear in width and H=200 extrapolates well
past our widest measured point. **2,000 GPU-h is too much to commit on an extrapolation.**

**Smoke first — 4 cells x 5,000 episodes, 1 seed, in parallel. ~8 GPU-h, ~3 h wall.**
Gates, all must pass before the real sweep:

```
   ess.jsonl   scored_frac >= 0.99          <- the actual claim, 100% coverage
               score_n_zero/scored_n reported per cell
               -> this is what settles H=200: is it really ~89% zeros at full coverage?
   timing      real hours/run, replacing the 34-152h extrapolation
   memory      no OOM at the chosen cf_score_chunk
   eval.jsonl  exists; win_rate + crash_rate + timeout_rate == 1.0
   coll_rew    the -1 cells show crash_rate affecting return (baseline sanity)
```

Then size the real sweep from measured numbers, `--dry-run`, submit, Monitor.

## Verification

1. `score_on_insert=False` reproduces an existing run bit-for-bit (default-off contract).
2. Smoke gates above.
3. After the sweep: `collect(_label)` must still return uniform n=26 / per n=18. If either
   grows, the `coll_rew` guard is not working.

## Explicitly not doing

No capacity change (100,000, settled). No `beta` sweep. No step-1 targeted-sampling arm —
score-on-insert supersedes it. No new metric or aggregation.
