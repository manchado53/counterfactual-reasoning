# Plan — supply-normalize precision@k (fix the visitation confound)

## Problem
`precision@k(draws)` is visitation-confounded: raw draw counts ≈ how often the agent VISITS a
state (buffer composition), not whether the priority preferentially DRILLS it. On FL-det,
Spearman(draws, stakes) ≈ -0.65 (most-drawn = low-stakes start states). All algorithms look
alike → the metric can't separate CCE from PER/uniform. Confirmed on the 21 completed runs.

## Revisions after 3-agent review (2026-06-20)
- Headline stat = **Spearman(oversampling, stakes)**, NOT precision@k (heavy-tailed ratio → use
  ranks). precision@k kept secondary, reported with a `k/S` random baseline.
- Noise mitigation = **Laplace shrinkage** (α ≈ mean adds/state), not a hard min-supply floor
  (a floor re-introduces the visitation bias by discarding rare high-stakes states).
- Buffer DOES wrap (capacity 100k ≪ millions of transitions) → cumulative adds ≠ occupancy.
  Log **adds-snapshots** and compute oversampling **per training window** (and a pre-convergence
  window), where within-window adds ≈ occupancy.
- Fix snapshot labeling bug: snapshots fire on q_update_count but are labeled by episode
  (colliding labels). Record q_update_count as the snapshot index.
- **FL-stoch is PRIMARY** (real C(s) gradation → CCE priority diverges from PER). FL-det CANNOT
  separate CCE from PER (degenerate score, CCE≈PER by construction) → use FL-det ONLY for the
  sane-vs-stale delivery control; do NOT report CCE-vs-PER precision on FL-det. Gate per run on
  DISTINCT-TD; if ≈0, report "indistinguishable by construction."
- Pass criterion (FL-stoch): CCE mean-oversampling on top-10%-stakes states ≥ 1.3× PER's with
  non-overlapping 3-seed bands; uniform ≈ flat (≈1) → precision@k ≈ floor.
- NEXT STEP (not this change): log the buffer's INTENDED priority per state directly →
  Spearman(priority, Q*) (skips sampling noise) + KL(realized‖intended) as the collapse alarm.
  This is the cleanest metric; defer to keep this change focused.

## Fix (one idea)
Log **supply** (adds per state) alongside **draws** (already logged), then score on the
over-sampling ratio:

    draw_prob(s)   = draws(s) / total_draws
    supply_prob(s) = adds(s)  / total_adds
    oversampling(s) = draw_prob(s) / supply_prob(s)     # >1 = drilled beyond its fair share

precision@k and Spearman then use `oversampling` (vs stakes) instead of raw draws. This cancels
visitation and leaves only "did the priority over-weight high-stakes states."

## Known risk to handle (the dual confound)
Dividing by a tiny supply makes rarely-added states explode to huge spurious ratios. Mitigate:
- min-supply floor: only rank states with adds(s) ≥ some threshold (e.g. ≥ a small % of total),
  OR Laplace/shrinkage smoothing on the ratio. Report how many states were excluded.

## Code changes
1. **`shared/draw_log.py` (DrawLogMixin)** — add an adds tally mirroring draws:
   - `_init_draw_log`: add `self._add_counts = {}`.
   - new `_record_add(self, transition)`: increment `_add_counts[(s,a)]` (gated by enable_draw_log).
   - `dump_sampling`: also densify + save `adds` (and `adds_snapshots` for drift symmetry).
2. **`shared/consequence_buffers.py` `add()`** and **`shared/buffers.py` `add()`** — call
   `self._record_add(transition)` (add_batch routes through add(), so it's covered). Cheap dict
   increment, gated → zero cost when off.
3. **`analysis/suitability/rollout_sweep.py: compute_realized_sampling`** — load `adds`, compute
   `oversampling`, apply the min-supply floor, run precision@k + Spearman on it. Keep raw-draws
   precision@k too (for the before/after contrast). Handle old npz lacking `adds` gracefully
   (fall back to raw + flag `supply_normalized=False`).
4. **`analysis/suitability/run_realized_sampling.py`** — surface oversampling precision@k in the
   figure/JSON (the headline), raw as secondary.

## Rerun
The 21 existing `sampling.npz` lack `adds`, so rerun both sweeps after the code change:
`precision_at_k_contrast` (12) + `precision_at_k_stoch` (9). FL is fast.

## Verify
- smoke: tiny run → npz has `adds`; oversampling well-defined.
- sanity: uniform run should give oversampling ≈ flat (≈1 everywhere) → precision@k ≈ random floor.
  PER/CCE should lift oversampling on high-priority states. CCE-sane >> CCE-stale if the fix works.
- the headline question: does CCE oversample high-STAKES states more than PER does?

## Open caveat (not fixed here)
On FL-det the consequence score is degenerate (~4% of states have signal), so CCE's priority ≈ PER
there anyway — FL-stoch (real C(s) gradation) may be the more informative env for this metric.
