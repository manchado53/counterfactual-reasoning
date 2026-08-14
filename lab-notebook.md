# Lab Notebook — Counterfactual Reasoning (CCE)

Read first. LOG = append-only (never edit). Rewrite only STATUS/NEXT.
Before /clear: append a dated LOG entry, especially DEAD ENDS.

## GOAL
ICLR 2027 (iclr.cc). Deadline ~late Sep 2026 (est.; recheck). ~16 weeks from 2026-06-02.

## THE BAR
  C2 (speeds learning):   have FL-det ✓  → need 2 MORE clean scenarios
  C1 (finds the moments): need deterministic-FL redo + 1 MORE (needs a good-oracle env)

## COVERAGE
```
            C1 (finds moments)       C2 (speeds learning)
FL-det      REDO (was slippery)       WIN ✓ mul 1.00 vs PER 0.46  (25 seeds)
FL-stoch    noise kills score         NULL 0.67–0.75              (10 seeds)
SMAX-3m     no oracle                 MARGINAL 0.722 vs 0.710     (10 seeds)
SMAX-8m     —                         UNFINISHED (never ran clean)
C4          —                         NOT FAIRLY TESTED (buggy)   <- DIG
Chess       oracle too weak           no improvement              DROPPED
```

## STATUS (2026-06-02)
Data so far = verified + frozen in paper/repro/ (master 861c0e3).
NOT done experimenting — no claim hits its target scenario count yet.

## NEXT
1. C4 Layer 1 — verify fixes are in code, then: does plain DQN beat the opponent at all?
2. Redo C1 on deterministic FL.
3. Fix 3 paper.tex numbers (FL-det → 25 seeds; FL-stoch → 0.67–0.75; SMAX PER → 0.71).
4. Recheck ICLR 2027 deadline.
5. [side track, not a paper scenario] JaxNav robotics-transfer (branch
   worktree-research+cce-robotics-transfer): 25-seed/150k rerun of the 3 arms
   (cce-max/cce-wmean/per) IN FLIGHT as of 2026-08-13. Job IDs 272116-272140 (cce-max),
   272141-272165 (cce-wmean), 272166-272190 (per); manifest
   `agents/jax_nav/experiments/holes_25seed_150k/manifest.json`. Fresh runs, not resumes.
   Check with `sacct -j <jobid> -o JobID,State,Elapsed -X`. PER finishes ~3x faster than the
   CCE arms (no counterfactual rollouts). When all 75 are done:
   `PYTHONPATH=<worktree>/src python -m counterfactual_rl.analysis.claim2.jaxnav_holes_figures`
   — `fig_25seed_150k` already exists and reads the manifest; nothing to add.
   **CORRECTION to the 2026-08-13 LOG entry (that entry is append-only, so it stands as
   written — this is the fix).** Its tail-trend read, "PER flat (+0.3pp), CCE+max rising
   (+4.2pp), CCE+wmean falling (-4.3pp)", is an artifact of the estimator, not a finding. It
   used last-minus-first of a trailing window, which is dominated by endpoint noise and flips
   sign with the window (PER: -1.2pp at 3k, +15.9pp at 20k on the same data). Refitted per
   seed with a CI across seeds (commits 3d5f2c8, bae75a7), the 96k truth is that ALL THREE
   arms were still climbing: PER +11.8pp [+5.5,+18.2] over its last 20k, CCE+wmean +10.3pp
   [+1.4,+19.3], CCE+max +7.3pp [-4.7,+19.3]. So 96k undershot for every arm, PER included —
   the relaunch was right, its stated reason was not. Do not cite "PER had converged at 96k".
   Early read of the 150k run (17 PER seeds already at full budget): PER 62.0% mean / 64.2%
   IQM vs 60.3% / 62.7% at 96k, trend +1.9pp [-9.6,+13.5] = consistent with settled. So 150k
   buys PER ~2pp and does look like a plateau; the CCE arms decide the comparison.
   Analysis now refuses to hide a thinned arm: it prints per-arm coverage and excludes seeds
   that did not reach 95% of the manifest's budget (before, a seed that died at 40k silently
   contributed its ep-40k score to a "150k final" mean, and truncated the whole IQM curve).

## PRE-FLIGHT CHECKLIST (run for every env — these broke us before)
- [ ] rewards[agent_player], not rewards[0]
- [ ] target/eval/save = crossing logic (a//f > prev//f), not `% f == 0`
- [ ] LeakyReLU on sparse/boolean inputs (ReLU → dead net)
- [ ] cf_horizon = full episode length
- [ ] consequence_metric = total_variation (not wasserstein)
- [ ] sanity: plain DQN must beat the opponent BEFORE testing CCE

## DEAD ENDS (ruled out — don't redo)
- Chess C2: no improvement.  Chess C1: oracle player too weak.
- SMAX C2 (3m marginal): no real gain.

## GOTCHAS (facts that bit us)
- runs live in agents/<env>/runs, NOT shared/runs.
- /home is a shared disk at 100% (294 GB free of 102 TB). Writes can fail and large
  uncommitted files are at risk. Keep anything important in git (that's what paper/repro/
  is for). [C1 deterministic runs vanished — cause unconfirmed.]

## IDEAS NOT TRIED
- slip-robust metric (spread of MEAN returns, not TV) — may rescue stochastic null.
- adaptive μ (CCE early → TD late).
- C1 perturbation/causal test.
- 2nd good-oracle env for C1 (small solvable MDP).

## SCOPE
Active: FL-det, FL-stoch, SMAX-3m, C4.   Dropped: Chess, raw diagnostics.

## LOG (append-only, newest on top)

### 2026-08-13 — JaxNav: found+fixed a real aggregation bug (issue #3), reran properly powered — mixed result, not a clean win
Branch `worktree-research+cce-robotics-transfer`. Follow-up to 2026-08-06's port. Two things
happened this session: a real bug fix, and a statistically honest re-test that came back murkier
than the earlier small-seed runs suggested.

**The bug (repo issue #3, confirmed present in the JaxNav port too, not just FrozenLake).**
`compute_consequence_metric(..., aggregation='weighted_mean')` only takes the weighted-mean
branch if `action_probs` is supplied (`analysis/metrics.py:306`); JaxNav's
`_score_buffer_transitions` never supplied it, so despite the config saying `weighted_mean`, the
code silently fell through to `max` over the 14 alternative-action divergences — every score was
"how different was the single weirdest alternative," not a real average. Same bug as FrozenLake,
carried over because the JaxNav port mirrored that file. **Fix**: `consequence_dqn.py` now
computes real `action_probs` = `softmax(Q(obs)/cf_rollout_temperature)` per scored state and
passes them through, so `weighted_mean` genuinely averages, weighted by how likely the policy
would have taken each alternative. Cheap pre-retrain check confirmed the fix changes the score
*distribution* (53 distinct values vs 18, no more ceiling-saturation) but barely reorders which
states get flagged on a late checkpoint (Spearman 0.997) — real effect, uncertain practical size.

**Also new**: a `PlateauEarlyStopper` (`agents/shared/early_stopping.py`) — smoothed, patience-based
early stop for runs where different arms converge to different final levels (the existing
`early_stop_win_rate` target-threshold stop doesn't fit that case). Wired into both vectorized
trainers. Left disabled (`early_stop_patience=100000`) for the runs below after an earlier
96k attempt (v3) showed it firing prematurely, right as exploration ended, on every single seed.

**Results, in the order run (holes map throughout — 8x8, 10% obstacles, `coll_rew=0`):**
```
12k episodes, 3 seeds/arm (quick fix-vs-bug check):
  CCE+weighted_mean(fix)  24.4%   CCE+max(bug, on purpose)  20.5%   PER  16.0%
  -> fix beat bug beat PER, but n=3, not powered.

96k episodes, 5 seeds/arm, OLD buggy aggregation (v4, no early-stop):
  CCE-mul  70.3% (IQM 70.1%)   vs   PER  56.1% (IQM 60.9%)
  Exact permutation test (all 252 splits): p=0.310 -- NOT significant despite the size of the gap.
  [Earlier in this session I mis-recalled this comparison from memory against a DIFFERENT run's
  numbers (v3, the buggy-early-stop one) and reported PER winning -- that was wrong. Re-verified
  against docs/figures/real/claim2/jaxnav/fig_jaxnav_iqm_v4.png, generated straight from
  runs/271903-271912/metrics.log. This is the corrected, code-verified number.]

96k episodes, 25 seeds/arm, power-analysis-justified (a Monte Carlo power sim off the 5-seed
stats above said ~25/arm needed for 80% power at this effect size and PER's variance):
  CCE+max(bug)        68.7%  (IQM 71.7%,  std 15.7%)
  CCE+weighted_mean(fix) 62.2%  (IQM 66.0%,  std 15.7%)
  PER                 60.3%  (IQM 62.7%,  std 16.4%)
  CCE+max vs PER:      t-test p=0.079,  Mann-Whitney p=0.012   (borderline)
  CCE+wmean vs PER:    t-test p=0.691,  Mann-Whitney p=0.491   (not significant)
  CCE+max vs CCE+wmean: t-test p=0.160,  Mann-Whitney p=0.051   (borderline)
  -> Honest read: CCE (either aggregation) is at least holding its own vs PER, never losing to
  it, but nothing here clears a real significance bar. The "fix helps" story from the 12k test
  did NOT reproduce cleanly at scale -- if anything the old buggy 'max' scored highest.
  Tail-trend check (slope of IQM over the last 5000 episodes): PER flat (+0.3pp), CCE+max still
  RISING (+4.2pp), CCE+wmean still FALLING (-4.3pp) -- none of the CCE curves had converged by
  ep 96k, unlike PER. Open decision: run longer (to let CCE settle) vs. more seeds (to shrink
  PER's own 16pp std, which is the main driver of the wide CIs) -- not yet chosen.

**Reminder on what "CCE" means in every number above**: `priority_mixing='multiplicative'`
(Eq5), `mu_c=mu_delta=1.0` (both default, never overridden). This is CCE+TD combined
(`p_C^mu_c * p_TD^mu_delta`), not a CCE-only arm. `mu` (the additive-mixing weight) is unused
here entirely -- multiplicative mixing never reads it.

**Infra**: `dh-node12` now confirmed bad (2nd incident — 3 co-scheduled jobs in the 25-seed run
all died SIGKILL at the same elapsed time). `--exclude=dh-node12` from now on.

Figures + generating code: `analysis/claim2/jaxnav_holes_figures.py` (extends a script found
already in the repo from earlier in this session, before a context summarization — same
job-ID-driven, no-hardcoded-numbers style) → `docs/figures/real/claim2/jaxnav/` (3 PNGs) +
`docs/figures/real/claim2/jaxnav/data/` (4 manifests, job_id → config). Rerun with
`PYTHONPATH=<worktree>/src python -m counterfactual_rl.analysis.claim2.jaxnav_holes_figures`.

Job ranges (Rosie, checkpoints not committed): v4 96k/5seed 271903-271912; 12k agg test
271925-271930; 12k max-control 271931-271933; 25-seed power run 271949-272023 (cce-max),
271974-271998 (cce-wmean, 3 replaced as 272041-272043 after the dh-node12 failure), 271999-272023
(per).

### 2026-08-06 — JaxNav robotics-transfer port: CCE wins on obstacles, loses without them
Branch `worktree-research+cce-robotics-transfer` (off `research/cce-robotics-transfer`), NOT
part of the paper's COVERAGE table — a side exploration into whether CCE transfers off grids
onto a continuous-state wheeled robot (JaxMARL's JaxNav). Mirrors the FrozenLake code
file-for-file: new `envs/jax_nav.py` adapter (calls `step_env`, not the auto-resetting base
`step`, so counterfactual rollouts branch from arbitrary stored states), new
`agents/jax_nav/{config,dqn,dqn_vectorized,consequence_dqn,consequence_dqn_vectorized,train}.py`,
a pytree-state fix in `shared/consequence_buffers.py` (`add_batch` now `jax.tree.map`s instead
of `int()`-casting — backward-compatible with FrozenLake's scalar states), and a `jax_nav`
branch in `analysis/claim2/parse_logs.py` + `run_analysis.py`.

Two JaxNav-specific things that mattered:
- JaxNav's discrete mode is 15 fixed `(v, ω)` motion primitives; single-agent transitions are
  **deterministic** (no slip like FrozenLake), so a greedy CCE rollout continuation collapses
  every action's return distribution to one point and TV degrades to 0/1. Fixed by a new
  `cf_rollout_temperature` knob (softmax-sampled continuation, default 0.5) — confirmed by a
  direct probe: greedy gave 2 distinct consequence-score values across 4096 scored transitions,
  temp=0.5 gave 6.
- Vanilla DQN was unstable on JaxNav (learned to 66% then collapsed to ~25%) until adding
  Double DQN + fixing the replay ratio (`n_steps_per_update` 4→16, `target_update_freq`
  200→2000) — both now defaults in `agents/jax_nav/config.py`.

Results (goal-reach rate = `state.goal_reached`, not `return>0`, so it's shaping-independent):

```
EASY map (6x6, no obstacles, goal_radius=1.0), 5 seeds, 8000 episodes:
  DQN+PER       77.7%  (reaches 60% threshold at 344k env-steps)
  DQN-Uniform   77.2%
  CCE-mul       72.1%  (385k steps to 60%) — PER/Uniform beat CCE here.

HOLES map (8x8, 10% obstacles, coll_rew=0 so a crash is a 0-reward terminal — JaxNav's
"hole"), 3 seeds, 24000 episodes, score_interval=500 (tuned so CCE's counterfactual scoring
doesn't dominate wall-clock):
  CCE-mul   37.3% final (per-seed: 23.8%, 29.4%, 58.6%)
  DQN+PER   18.3% final (per-seed: 14.6%, 26.0%, 14.2%)
  -> CCE ~2x PER. High seed variance (one CCE seed carries the mean) — a 5-seed/48k-episode
  rerun is in flight to firm this up.
```

**Verdict so far: same mechanism as FrozenLake.** CCE has no edge when the agent can always
recover (easy map); CCE wins when a mistake is irreversible (holes map). This is the
robotics-transfer analog of the FL-det headline result.

Job IDs (Rosie, `agents/jax_nav/runs/<job_id>/`, checkpoints NOT committed — regenerate via
the configs in `docs/figures/real/claim2/jax_nav/data/manifest_*.json`):
- Easy pilot (1 seed each): 271047 (CCE-mul), 271048 (PER)
- Easy 5-seed sweep: 271052-271056 (CCE-mul), 271057-271061 (PER), 271062-271066 (Uniform)
- Holes pilot (1 seed each): 271050 (CCE-mul), 271051 (PER)
- Holes 3-seed, 24k eps: 271070-271072 (CCE-mul), 271073-271075 (PER)

Figures + provenance JSON committed under `docs/figures/real/claim2/jax_nav/` (figures) and
`docs/figures/real/claim2/jax_nav/data/` (curve JSON + manifests — kept outside the gitignored
`experiments/` tree so the numbers survive even if Rosie's copies don't; see GOTCHAS below).

Reproduce: `PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src python -m
counterfactual_rl.agents.jax_nav.train --algorithm consequence-dqn --mixing multiplicative
--seed 0 --override sparse_reward=True --override 'map_size=(8,8)' --override fill=0.1
--override goal_radius=0.8 --override coll_rew=0.0 --override max_steps=200 --override
score_interval=500 --episodes 24000` (swap `--algorithm dqn` for the PER baseline). Sweep
driver: `slurm/sweep_holes_long.py` in the worktree.

### 2026-06-02 — repo + docs cleanup; notebook is now source of truth
- Stood up this lab-notebook + a CLAUDE.md "read first" rule (durable on-ramp for new agents).
- Verified all 3 paper datasets exist on disk: FL-det 124/125, FL-stoch 50/50, SMAX-3m 50/50
  (runs live in `agents/<env>/runs`; legacy SMAX in `agents/shared/runs`).
- Archived stale/contradicting docs → `docs/_archive/` (PAPER_STATUS, PAPER_CLAIMS, paper.md,
  CCE_DIAGNOSTICS, mock figure scripts, pre-pivot SMAC/multidiscrete docs). `docs/` is now active reference only.
- Rewrote CLAIM1/CLAIM2_METRICS as pure metric cookbooks (definitions only, no results).
- Rewrote CLAUDE.md as a high-signal durable reference (custom FrozenLake, training gotchas, CCE knobs, how-to-run).
- Disk: `/home` is a SHARED disk at 100% (not a personal quota; our footprint ~640 GB).
- Commits: 861c0e3 (repro bundle), cb0b9be (docs cleanup), 16c00dd + a7423bf (CLAUDE.md).

### 2026-06-02 — froze repro bundle + verified paper numbers
Built paper/repro/ (manifests + .npz caches + C1 checkpoints) + FIGURE_PROVENANCE.md. Commit 861c0e3.
Verified from cache:
- FL-det: mul 1.00, add 0.83, only 0.62, PER 0.46, uniform 0.31 (25 seeds)
- FL-stoch: null 0.67–0.75 (PER 0.749) (10 seeds)
- SMAX-3m: mul 0.722, PER 0.710, others 0.67–0.69 (10 seeds)

Repo scan: SMAX-8m never ran clean (analysis timed out); C4 bugs (reward-seat, dead-ReLU)
FIXED IN CODE but never re-run clean → C4 not fairly tested.
