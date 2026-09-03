# Handoff — instrument the buffer, then run (2026-09-03)

Written to hand this to a fresh session. `lab-notebook.md` (entry 2026-09-03) is the
source of truth for WHY; this file is the actionable subset with the context needed to
execute without re-deriving anything.

## Where to work

```
worktree   .claude/worktrees/research+cce-robotics-transfer
branch     fix/cce-rollout-bootstrap   (HEAD f6268cb)
python     ~/.conda/envs/counterfactual/bin/python    NOT the machine default
env vars   PYTHONNOUSERSITE=1   PYTHONPATH=$WT/src    (both required)
```

## The finding this run exists to confirm

Only ~1.6% of the JaxNav replay buffer ever receives a *measured* CCE score. The rest
carries the running mean it inherited on insertion (`consequence_buffers.py:102`, which
is what Algorithm 1 line 7 of the paper specifies). Capacity cancels:

```
                       n_score_sample
   E[times scored] = ---------------------------------
                     score_interval x n_steps_per_update

   JaxNav      64 / (250 x 16)  =  1.6%     -> C2 null
   FrozenLake 128 / (100 x  4)  = 32.0%     -> C2 win
```

**This is DERIVED, not measured.** Nothing in the buffer tracks which entries hold a real
score. Confirming or killing it is the single most important output of this run. Do not
present 1.6% as a measurement until `scored_frac` exists.

## Build: ~73 lines, no new algorithms

### 1. `agents/shared/consequence_buffers.py`

- Add `self.scored_at = np.full(capacity, -1, dtype=np.int64)` next to
  `consequence_scores` (line ~77).
- `add()` (line ~112): set `self.scored_at[pos] = -1`.
- `update_consequence_scores()` (line ~311): set `self.scored_at[idx] = q_update_count`
  (pass it in).
- `priority_diagnostics()` (line ~238) returns three more fields:
  - `scored_frac` = `mean(scored_at >= 0)`  ← settles the 1.6%
  - `staleness_p50/p90` = percentiles of `now - scored_at` over scored entries
  - `score_hist` = ~100 **log-spaced** bins over [1e-5, 1] **plus a separate
    `score_n_zero` counter** (a log axis cannot hold 0, and "exactly zero" is a real
    category). Linear bins are wrong here: `score_mean` travels 0.0014 -> 0.075 over a
    run, so linear bins put everything in bin 0 early and bin 7 late.
- New `snapshot_buffer(path)`: `np.savez_compressed` of `consequence_scores`,
  `td_magnitudes`, `scored_at`, `_write_pos`. ~1.2 MB. Call it at ~10 checkpoints.
  **This is the insurance** — every question this week needed a fresh GPU job because the
  buffer is never saved.

### 2. `agents/jax_nav/consequence_dqn.py`

- `_build_rollout_fn` (line ~76): also return the terminal reason per rollout
  (goal / crash / timeout / alive-at-H). The pattern already exists in
  `~/probe_logs/probe_reasons.py` — copy it, it mirrors `single_rollout` exactly.
  Aggregate the four shares into `priority_diagnostics` output.
- Double-sampling (Jeremy 07/10, never built): every N updates draw a second batch using
  **TD priority only**, do NOT apply it, log the overlap with the real batch. ~15 lines.
- Normalise-TD flag (Jeremy 08/20, never actioned): `normalise_td: False` — min-max or
  rank-normalise `td_magnitudes` to [0,1] before mixing. Measured motivation: TD carries
  **52-129x** the relative spread of CCE, and multiplicative mixing gives both the same
  exponent.

### 3. `agents/jax_nav/dqn.py` — `evaluate()` (line ~120) and `_MetricsLogger` (line ~60)

`evaluate()` already computes `goals` from `final_state.goal_reached`; the same state
carries `move_term` (crash) and `step`. Add:

```
   crash_rate, timeout_rate, avg_length_goal, avg_length_crash
```

`avg_length_goal` is Jeremy's 08/27 ask (length conditional on survival) — today's
`avg_length` averages crashes and successes together, so it moves for two opposite
reasons at once.

**Write these to a NEW `eval.jsonl`, not `metrics.log`.** `metrics.log` is a fixed-width
table every figure module parses; adding columns breaks them. Put the write in
`_MetricsLogger.log_eval` so **uniform and PER get it too** — crash rates have to be
comparable across arms, and `ess.jsonl` only exists for CCE runs.

## Run

```
   0. SMOKE FIRST: 1 arm, 5k episodes, ~10 min.
      Check every new field appears and is sane. The instruments are untested and a
      logging bug found at 6am costs the whole night.

   1. Overnight:  cell 8x8_f03
                  arms   uniform, per, cce_wmean, cce_wmean_bs, cce_wmean_tdnorm
                  seeds  3        (Jeremy 08/13: "three is the technical minimum")
                  length FULL 250k  -- required for the trajectory figures
                  15 runs, ~66 GPU-h, one wave at 30 concurrent, ~6.5 h wall
```

Driver: copy `slurm/sweep_bootstrap.py`, change ARMS and the manifest path. It already
handles the array, throttle, config base64 and manifest.

## Analysis owed — NO COMPUTE, do before the meeting

1. **P(method > baseline) via bootstrap.** Jeremy 08/13: *"even 80-90% would be a strong,
   reportable result"*, and *"one-tailed uses 90% confidence; two-tailed uses 95%"*. We
   have been reporting two-tailed p-values instead. Recompute from array 274476 on disk.
2. **Compute-time analysis.** Jeremy 07/10, never done. `timing.jsonl` on every run has
   it. Measured: uniform 1.25 h, PER 1.72 h, CCE 12.8 h at 40 rollouts / ~6.4 h at 20.
3. **Fix panel 1 of `fig_why_null.py`** — remove the two grey reference bars (0.60, 0.47).
   They came from a different sweep at a different exponent and compare a knob setting,
   not a signal.

## Figures the run should produce

Mockups with synthetic data are committed at
`docs/figures/real/claim2/jaxnav/MOCKUP_planned_figures.png` (watermarked). Panels:

```
   A  buffer score distribution over training   heatmap; Jeremy's KDE ask (#8)
   B  score vs STALENESS                        tests the paper's own staleness claim
   C  how the ROLLOUTS end                      goal/crash/timeout/truncated
   D  how the EVAL episodes end                 replaces a crash BOUND with the number
   E  length conditional on outcome             Jeremy 08/27
   F  what actually got replayed                needs the sampling.npz port -- SKIP for now
   G  CCE vs TD spread over training            Jeremy 08/20
   H  double-sampling, CCE vs PER batches       Jeremy 07/10
```

C and D together are the highest-value pair: CCE assumes the rollouts predict reality and
**nobody has ever checked**. ~14 lines.

Log the histogram EVERY EVAL, not once. Same data then gives the heatmap (paper), a
5-checkpoint ridgeline (slide) and an animation (talk) with no re-run.

## Gotchas that have cost real time

```
  SLURM array tasks write to JobIDRaw, not <array>_<task>.
     274476_10 lands in run dir 274487. Resolve via sacct before any analysis.
  /tmp is NODE-LOCAL. A script there fails under sbatch in ~1s with exit 2,
     which looks like a code error. Put scripts in $HOME or the worktree.
  PYTHONNOUSERSITE=1 is required -- ~/.local has a broken cffi that breaks jaxmarl.
  PYTHONPATH must put the WORKTREE first; the editable install points at the main repo.
  score_zero_frac in ess.jsonl is MISLEADING. New entries inherit the running mean,
     never 0, so it reads ~0 regardless. Use score_std/score_mean.
  dh-node12 is a confirmed bad node -- exclude it.
  A benign JaxNav FutureWarning (scatter int32->bool) appears in every run. Ignore.
```

## Do NOT

```
  - sweep beta. (0+eps)**beta is one constant for 98.4% of the buffer.
    NEXT item 0 used to say the opposite; it was wrong.
  - run the coverage sweep before scored_frac exists. Its premise is still derived.
  - present 1.6% as measured.
  - try to rescue JaxNav with more seeds. Paired sd is 30pp; detecting 5pp needs
    ~290 seed-pairs. The environment cannot resolve the effects we care about.
```

## Context a fresh session will want

```
  issues     #7 bootstrap (done)  #8 KDE  #10 sparse-vs-shaped  #11 bootstrap-vs-cap
             #12 rollout duplicated x5    #13 FL-det binary score   #15 Algorithm 1 gate
  meeting    Notion "09/03 Touchpoint", 5 bullets, in Jeremy Research/Meeting Notes
  deadline   ICLR abstract Sep 18 -> 15 days. Paper has ONE clean C2 win (FL-det).
  paper      paper/Counterfactual_RL + Theory.pdf (2026-08-13) is CURRENT.
             paper/paper.tex is STALE (2026-06-02). Reconcile before editing either.
```
