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
FL-det      REDO (was slippery)       WIN ✓ 18/20 seeds vs PER 10/20 (20 seeds, 08-04)
FL slip>0   —                         NULL at every level 0.02-0.133 (20 seeds) <- KNIFE EDGE
FL-stoch    noise kills score         NULL 0.67–0.75              (10 seeds)
SMAX-3m     no oracle                 MARGINAL 0.722 vs 0.710     (10 seeds)
SMAX-8m     —                         UNFINISHED (never ran clean)
C4          —                         NOT FAIRLY TESTED (buggy)   <- DIG
Chess       oracle too weak           no improvement              DROPPED
```

## STATUS (2026-08-04, evening)
Graded-slip dense sweep DONE for the low-slip window (560 runs, 0 failures). Verdict:
CCE-mul beats PER at slip=0 ONLY — a knife edge, not the graded decay Theorem 3 predicts.
Honest claim now: CCE's replay benefit is SPECIFIC TO DETERMINISTIC environments.

Data so far = verified + frozen in paper/repro/ (master 861c0e3).
NOT done experimenting — no claim hits its target scenario count yet.
Graded-slip work lives on `experiment/graded-slip-frozenlake` (worktree
`.claude/worktrees/graded-slip-frozenlake`), NOT yet merged to master.

## NEXT
0. **Decide the paper's framing given the knife edge.** The graded-slip result narrows C2 to
   deterministic envs. Either (a) own it — "CCE is a determinism-specific replay signal", and
   make the slip sweep the evidence, or (b) keep hunting for a 2nd env where it holds. Talk to
   Jeremy: Theorem 3 predicts a slope we did not find, so the theory needs revising either way.
   DO NOT spend the remaining 683 runs (0.20-0.666 are known ties; probe is answered).
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

### 2026-08-04 (later) — DENSE SWEEP ANSWERS IT: **KNIFE EDGE, not a decay.** Thm 3 slope NOT supported
The submitter was killed at 592/1280 (the parent process exited; manifest never written — recovered
from the submit log via `recover_manifest`). **All 592 COMPLETED, zero failures** (dh-node12 exclude
holds; yesterday's rate was 7%). What landed is the 7 dense low-slip levels at the FULL 20 seeds:
slip 0.0/0.02/0.04/0.06/0.08/0.10/0.133 x 4 arms x 20 = 560 runs. That is exactly the window all the
signal lives in, so the question is answered even though the tail never ran.

**USE ESCAPE RATE, NOT IQM, IN THIS REGIME.** IQM reads 0.000 across slip 0.02-0.10 while the MEAN
win rate is ~0.25 — outcomes are bimodal (seed solves or dies), so IQM's middle 50% is all zeros and
it reports a floor that is not there. Trajectories are PLATEAUED by 60-80% of training, so a bigger
episode budget would NOT rescue it (I proposed that first; it was wrong). Right metric = fraction of
seeds solved, i.e. the 07-05 basin-escape framing.

SEEDS SOLVED (final-10% mean win rate >= 0.5, out of 20), + bootstrap 95% CI on mul-PER:
```
  slip    uniform  PER  CCE-only  CCE-mul   mul-PER        CI
  0.000     10     10     13        18       +8/20   [+0.15,+0.65]  <- ONLY significant level
  0.020      1      5      0         5        0      [-0.25,+0.25]
  0.040      2      8      1         4       -4      [-0.45,+0.10]
  0.060      0      6      1         8       +2      [-0.20,+0.40]
  0.080      1      8      0         7       -1      [-0.35,+0.25]
  0.100      2     13      1        11       -2      [-0.40,+0.20]
  0.133      6     14      7        17       +3      [-0.10,+0.40]
```
**CCE-mul beats PER at slip=0 and NOWHERE ELSE.** Gone by slip 0.02 and it never comes back.

WHY THE NULL IS TRUSTWORTHY (not a power failure): the SAME measurement cleanly separates uniform
from PER at those very levels (uniform 0-2/20 vs PER 5-13/20). The instrument discriminates fine; it
just finds nothing between mul and PER. That is a real null.

CONSEQUENCES:
- **Theorem 3's smooth slope is NOT supported.** The advantage is not graded in noise; it is a cliff
  at exact determinism. Entropy at slip 0.02 is only H=0.112 nats and the advantage is ALREADY dead.
- The honest claim narrows to: **CCE's replay benefit is specific to deterministic environments.**
- The slip 0.8/0.9/1.0 PROBE is now near-pointless: it was meant to separate "needs LOW noise" from
  "needs DETERMINISM", and H=0.112 killing the effect already answers that — determinism. Probe
  levels sit at H=0.69-0.95, far noisier. Don't spend the 240 runs.
- **CCE-only collapses with any noise** (0-1/20 at slip 0.02-0.10, at or below uniform) while
  CCE-mul tracks PER. The TD half of the mixture is carrying the method off determinism.
- Deterministic anchor got STRONGER at 20 seeds: mul IQM 1.00 [1.00,1.00], 18/20 seeds solved,
  P(mul>PER)=0.70 [0.62,0.75]. Best FL-det result to date.

NOT RESUBMITTED: the remaining 683 runs (0.166 partial, 0.20-0.666, probe). 0.333+ is already known
ties on 10-seed data and the probe is answered. Figures: `docs/figures/graded_slip_dense/`.
Manifests: `experiments/2026-08/claim2_graded_slip_dense_2026-08-04/{RECOVERED_partial,COMPLETE_LEVELS}.json`.

### 2026-08-04 — graded-slip: found a silent seed-contamination bug, launched the DENSE sweep
Picked up the 08-03 sweep (200 runs, 5 slip levels, 10 seeds). It DID run: 186 completed,
14 killed, 48.7 GPU-h, 2 h wallclock.

TWO BUGS, both fixed and committed (`experiment/graded-slip-frozenlake`, d2e91d3 + 9e50680):
- **All 14 kills were on ONE node, dh-node12** (SIGKILL, batch step CANCELLED, ~1 GB of 32 GB
  used — not OOM, not the 14 h limit, and PreemptMode=OFF cluster-wide so not preemption).
  The other 186 runs spread over 16 nodes with zero failures. Added dh-node12 to the sbatch
  `--exclude` (it now joins 16/17/18). CHECK THIS FIRST when runs die for no reason.
- **`parse_logs.load_manifest` forward-fills short seeds** — correct for an early-stopped
  winner (fill ~1.0), WRONG for a killed run (fill at its dying win rate, often 0.0). So the
  14 dead runs were counted as seeds that never learned. New `parse_logs.filter_complete_runs`
  (complete = reached episode budget OR hit early_stop_win_rate) drops them; validated because
  it recovers exactly the 14 jobs sacct calls non-COMPLETED, from logs alone. ADDITIVE — the
  frozen paper pipeline is untouched. `graded_slip.py --keep-incomplete` reproduces the old numbers.

WHAT THE 08-03 DATA ACTUALLY SAYS (clean, dead seeds dropped — contamination moved the headline
by <=0.05, so the original read STANDS):
```
  slip      0.0    0.166   0.333    0.5    0.666
  P(mul>PER) 0.65   0.43    0.41    0.34    0.55     <- only slip=0 clears 0.5
  IQM mul    1.00   0.98    0.96    0.88    0.71
  IQM PER    0.67   0.99    0.97    0.93    0.66
```
Theorem 3 wanted a smooth ramp. We got ONE moving point and four ties. CCE-only moved most
under the fix (IQM +0.13 at slip 0, +0.16 at 0.166).

WHY THE TIES ARE PARTLY AN ARTEFACT: **slip HELPS exploration** (random slides stumble into the
goal), so by slip 0.333 every arm ceilings at ~0.96 and final IQM physically cannot show a gap.
Between-arm spread is 0.67/0.84 at slip 0.0/0.166 but 0.01/0.08 at 0.333/0.5. All measurable
signal lives in **slip 0.0–0.25**.

NEW FINDING (suggestive, NOT a claim): steps-to-threshold sees under that ceiling. Median env
steps to a 0.5 win rate — CCE-mul is FASTEST at three levels final-IQM calls ties:
slip 0.166 164k vs PER 197k · 0.333 98k vs 131k · 0.666 66k vs 82k. But at the 0.9 threshold
the ordering is mixed (mul SLOWER at 0.166 and 0.666). Shape = **CCE-mul gets off the ground
faster, then the edge washes out** — consistent with the 07-05 BASIN-ESCAPE reframe. Caveats:
steps_to_threshold has median+IQR and NO bootstrap CI, 8–10 seeds, coarse eval grid.

LAUNCHED: `claim2_graded_slip_dense` — 16 levels x 4 arms x **20 seeds** = 1280 runs, ~250 GPU-h.
Dense over 0.0–0.25 where arms still separate; sparse above 0.333. 20 seeds because outcomes are
near all-or-nothing and the best 10-seed CI was P=0.65 [0.55,0.70] — too wide to resolve a slope.
PLUS a **probe at slip 0.8/0.9/1.0**: outcome probs are [p/2, 1-p, p/2], so entropy PEAKS at
p=2/3 (ln3=1.099) and FALLS after (p=1 -> [.5,0,.5], ln2=0.693). Past the peak slip rises while
noise drops — the one place "CCE needs LOW NOISE" and "CCE needs DETERMINISM" predict opposite
things. Probe caveat: past 2/3 the intended action is the least likely outcome, so if every arm
collapses the probe is uninformative rather than evidence.

GOTCHA THAT COST US THROUGHPUT (fix before the next sweep):
`train_frozen_lake_dqn.sh` asks `--cpus-per-gpu=32`, but a teaching node has **72 CPUs and 4 T4s**.
Two jobs exhaust the CPUs, so only 2 of 4 GPUs per node are ever used and concurrency caps at
**32 no matter what `--max-concurrent` says** (60 was requested; 32 ran). Halving the CPU request
to 16 should roughly double throughput. NOT changed mid-sweep — consistent resources across all
1280 runs beats the saved hours.

NEW FIGURES (`analysis/claim2/graded_slip.py`):
`fig_iqm_vs_slip` · `fig_advantage_vs_slip` · **`fig_advantage_vs_noise`** (x = entropy, not slip;
probe points drawn detached because p -> H is not injective) · **`fig_steps_to_threshold`**
(log y, marks points where most seeds never reached the bar). Clean 08-03 rerun lives in
`docs/figures/graded_slip_clean/`.

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
