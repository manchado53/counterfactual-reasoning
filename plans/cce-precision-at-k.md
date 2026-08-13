# Option B — precision@k + ESS (realized replay sampling)

## What we're doing
Make the trainer record which transitions it actually draws from the replay buffer during a
real run, then add an analysis that scores those draws:
- **precision@k** — of the transitions the trainer drilled most, how many are truly high-stakes?
- **ESS** — did the sampling collapse onto a handful of transitions? (alarm only)

This closes the loop the v1 (Option A) metrics can't see: Option A grades the CCE *score*;
precision@k grades what the score actually *did* to replay.

## Why
A clean score can still drill the wrong transitions if the pipe between score and sampling
leaks — stale scores (misset `score_interval` vs buffer turnover), priority mixing washing out
the signal, etc. precision@k is the only metric that would catch that; all 6 current metrics
would wave it through. It's also the "primary sanity" check named in the cookbook.

## How (sketch)
1. **Instrument the buffer** (`shared/consequence_buffers.py:sample`): behind a flag
   (`log_sampling`, default off — zero cost when off), accumulate a draw histogram keyed by
   the transition's STATE (FL state is a single int, so a 1-D count over states is enough),
   plus a global ESS accumulator. No per-step disk writes.
2. **Trainer dumps it** at run end (`consequence_dqn*.py`): write `sampling.npz` (draw counts
   per state, total draws, snapshot episode) into the run dir, next to `metrics.log`.
   Wire a config knob through `config.py` so sweeps can turn it on.
3. **Analysis** (`analysis/suitability/`): `compute_realized_sampling()` reads `sampling.npz`,
   builds `precision@k` against true stakes (exact Q*-spread for FL; C(s) elsewhere) and `ESS`.
   Surface both in the scorecard JSON + the deferred dashboard panel (currently n/a).
4. **Validate**: one short FL run with the flag on; confirm precision@k is high when
   `score_interval` is sane and drops when we deliberately misset it (the failure it's meant
   to catch).

## Scope / open questions (for plan mode)
- Vectorized vs non-vectorized trainer: CONFIRMED both share one `ConsequenceReplayBuffer`;
  the vectorized trainer overrides only adding, not sampling. So instrumenting `buffer.sample()`
  is a SINGLE point that covers both. Good.
- Buffer is a ring (slots reused on turnover) → key the histogram by transition identity
  (state), NOT raw slot index, so turnover doesn't corrupt counts.
- Generality: FL keys by state int; games (C4/SMAX) need a transition-id scheme (defer if so).
- Branch: we're on `cce-buffer-diagnosis` (fits this task) but it carries a big uncommitted
  suitability pile — decide whether to commit/clean that first.
