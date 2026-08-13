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
1. **CCE suitability predictor** — v1 pipeline BUILT + validated (`src/counterfactual_rl/analysis/
   suitability/`, cookbook `docs/SUITABILITY_METRICS.md`). Follow-ups (parked here so they're recoverable):
   - **Run the SLIP SWEEP** (the hero experiment). Env is unblocked: `slip_probability` added to
     `frozen_lake.py` (+`dqn.py` passthrough). Need cluster training across slip∈{0,0.05,…,0.33},
     then plot SNR↓ vs CCE-benefit↓. `qstar_spread_exact` already does weighted VI so GAIN-fidelity
     stays correct at any slip level.
   - **precision@k / ESS (Option B)** — needs trainer instrumentation of realized replay draws;
     formulas parked in the cookbook's "Deferred" section.
   - **Connect Four / SMAX adapters** — `suitability/envs.py` interface is thin; add
     `make_*_adapter` (no exact Q* → GAIN-fidelity = n/a).
   - Run on full det/stoch runs (not the single partial ckpt used to validate) for real numbers.
2. C4 Layer 1 — verify fixes are in code, then: does plain DQN beat the opponent at all?
3. Redo C1 on deterministic FL.
4. Fix 3 paper.tex numbers (FL-det → 25 seeds; FL-stoch → 0.67–0.75; SMAX PER → 0.71).
5. Recheck ICLR 2027 deadline.

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

### 2026-07-05 — RECONCILED the FL-det headline; reframed the win as BASIN ESCAPE
Chasing "why does CCE win in deterministic FL". Cleared up a metric confusion that had us
mis-stating the result, and reframed the whole question.

THE FIGURE vs THE RAW TRUTH (both correct, two summaries of the SAME data):
- Paper fig `paper/figures/fig1_iqm_frozen_lake_no_slip.png` y-axis = **IQM win rate** (rliable:
  drop top 25% + bottom 25% of seeds, average the middle 50%). Plateaus = 1.00/0.83/0.62/0.46/0.31
  (mul/add/only/PER/uniform) — these ARE the notebook's old shorthand numbers.
- **FL-det no-slip is BIMODAL per seed**: every seed ends win-rate 1.0 or 0.0, ZERO middling
  (verified from `paper/repro/cache/claim2_frozen_lake_no_slip.npz`, raw_0..raw_4, 25 seeds).
- So each arm differs ONLY in **how many seeds escape the dead basin**:
  uniform 10/25, PER 12/25, CCE-only 14/25, add 16/25, **mul 20/25**. frac-solved =
  0.40/0.48/0.56/0.67/0.80.
- **"mul = 1.0" is IQM, NOT "100% of seeds solved."** IQM trims the 5 dead mul-seeds → middle is
  all 1.0 → plateau reads 1.0. Raw success = **80% (20/25)**. HONESTY FLAG for the paper: plain
  success rate is 80%, IQM hides the 5 dead seeds. (IQM is standard/defensible, just know the raw #.)

REFRAME (the sharp version of the mystery):
  NOT "why does CCE learn faster/prioritize better" → it's all-or-nothing.
  TRUE Q = **"why does CCE's replay recipe flip MORE seeds ○→● (escape the dead basin)?"**
  In sparse det FL a seed either bootstraps goal-reward all the way back to start (→1.0) or never
  catches the thread (→0.0). CCE tips more coins toward escape. Winners/losers = CLEAN binary labels.

MECHANISM EVIDENCE so far (PRELIMINARY — see caveat):
- CCE and PER **do** sample differently: TV(draw distributions) = 0.517 on FL-det (CCE differs from
  PER MORE than either differs from uniform). So "same draws → same learning" is ruled out.
- vs oracle Q*-spread stakes (value iteration on env.P): cumulatively CCE looked LESS aligned than
  PER (0.15 vs 0.37 Spearman), but TIME-RESOLVED, CCE spikes to 70% of draws on top-stakes squares
  right at the solve vs PER's ~56% smeared over 3× longer. Candidate mechanism = sharp late
  concentration spike. NOT yet a claim.

CRITICAL CAVEAT (why the above is not paper-grade yet):
- Draw-logging (Option B, `shared/draw_log.py`) was added **Jun 20, AFTER the paper froze**. So the
  ONLY runs with draw logs are the **Jun 20 262xxx re-runs** (3 seeds, instrumented, WEAKER/mushier
  gap: mul~62% / PER~54% / uniform~48%). The **paper runs (257xxx, May, 25 seeds) have NO draw logs.**
  Everything mechanistic above is read off the 3-seed re-runs, single-seed cuts, unaligned axes.
- To study basin-escape properly we need draw logs on MANY seeds (enough winners AND losers). Plan:
  re-run the exact 5 paper arms (257xxx config: 8x8, no-slip, mu 0.25, 15000 ep) with
  `log_sampling=True` at ~25 seeds, confirm it reproduces the 10→20 escape ladder, THEN run the
  definition-free winners-vs-losers analysis (do future-escapers' pre-solve sampling cluster apart
  from future-dead seeds? — no square-labeling needed, outcome defines the target).
- TODO before that cluster job: verify draw-logging is side-effect-free (gated, no extra RNG draw)
  so the re-run matches the paper.

### 2026-06-14 — BUILT + validated the CCE suitability pipeline v1 (Option A)
Shipped `src/counterfactual_rl/analysis/suitability/` (envs, rollout_sweep, metrics, scorecard,
run_suitability, _smoke) + cookbook `docs/SUITABILITY_METRICS.md`. Env change: `slip_probability`
added to `envs/frozen_lake.py` (None = legacy behaviour, verified no regression) + `dqn.py`
passthrough. **Conda env = `counterfactual`, NOT bucks** (bucks has no jax).
Smoke + end-to-end run (run 255153, partial ckpt) both pass. Reusable: agent `_compiled_rollout_fn`,
`compute_consequence_metric`, value-iteration oracle, `gini`/`spearman_boot`.

TWO KEY FINDINGS (real data, preliminary):
- **CCE's total-variation consequence score is DEGENERATE in DETERMINISTIC envs.** Deterministic
  rollouts → point-mass return distributions → TV is coarse 0/1; only ~4% of det states even have a
  C(s) signal vs ~47% in slippery. So the GAIN-fidelity bridge (CCE vs exact Q*) is measured on
  **FL-stoch** with a slippery-trained policy (got ρ≈0.52), CONSISTENT with the old Claim-1 ρ on
  slippery FL. Implication: the distributional consequence score *needs* stochasticity to be graded.
- **SNR cleanly separates det vs stoch.** Must use ratio-of-AGGREGATES, not median-of-per-state-ratios
  (median→0 because most states are dead zones). Result: FL-det SNR≈1000 (within-action var=0, clean)
  vs FL-stoch SNR≈0.05 (env noise dominates the action signal). Strong quantitative support for
  "noise kills CCE in stochastic FL" (the FL-stoch NULL).
DISTINCT-TD ran 0.44–1.0 (CCE ≠ TD → go/no-go passes). Preliminary outputs:
`docs/figures/suitability/{scorecard.json,scorecard.png,dashboard_real.html}` — NOT paper numbers
(single old ckpt, reduced rollouts). Plan: `~/.claude/plans/mossy-pondering-frog.md`.

### 2026-06-14 — NEW THREAD: CCE suitability predictor (cheap "will-CCE-help?" metrics)
Idea worth trying: instead of one more env result, build CHEAP rollout-based metrics that
forecast/debug whether CCE helps in ANY env — transferable, run before/while spending compute.
These are DECISION TOOLS for us (what to pursue/avoid), not necessarily paper figures.

Organizing theory (Mattar & Daw 2018): worth replaying = GAIN × NEED. CCE estimates GAIN;
the metrics are conditions on GAIN + the NEED it forgets. The atom: roll out a policy, for each
state try each action (K rollouts), m(s,a)=mean return, C(s)=spread over actions ("stakes").

The locked metric set (after lit review + 3 adversarial review agents):
- GAIN real & visible: **Concentration** (Gini/top-k mass of C(s)), **SNR** = between-action var /
  within-action var with GREEDY rollouts (noise=env only) ≈ noise-normalized action gap.
- Beats PER: **DISTINCT-TD** = 1−|Spearman(CCE, |TD|)|. ★ THE GO/NO-GO — was MISSING; if CCE
  ranks like TD it's just slow PER. This is the single most important check.
- Reachable/visited: **NEED** = corr(C(s), successor-rep occupancy); **HORIZON-FIT** =
  cf_horizon / effective_horizon (Laidlaw 2023).
- Truth/sanity: **GAIN-FIDELITY** = Spearman(CCE, exact Q*-spread/EVB) — FL only, the calibration
  bridge; **precision@k** (primary) + **ESS** (collapse alarm only).
Review killed: circular Score-vs-C(s); demoted ESS; fixed SNR estimator (greedy, not ε-greedy).

Hero experiment (the real evidence, not a 4-env scatter): **SLIP SWEEP** — FrozenLake, slip
0.0→0.33, predict SNR↓ and CCE-benefit↓ together (dose-response). Sits between our det-WIN and
stoch-NULL anchors. Falsifies the thesis cheaply if flat.

Closest competitors to cite/differentiate: Yu et al. 2021 (Gain×Need for deep PER, arXiv:2111.14331);
Korniak 2026 (when non-uniform replay matters, arXiv:2605.10236 — found over-concentrated replay can
HURT → we measure top-k OVERLAP, not "concentration good").

DECISION: build **Option A** = rollout-only Tier-1 pipeline (no trainer edits), FrozenLake first,
warmup-sweep policy (random / ~10% / ~30% trained). Pipeline lives at
`src/counterfactual_rl/analysis/suitability/` (rollout_sweep, metrics, run_suitability, envs).
Plan: `plans/cce-suitability-predictor.md`. Mocks + interactive dashboard: `docs/figures/mock_preview/`.
NOT a Claude skill — it's research code. (Branch: research/cce-buffer-diagnosis.)

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
