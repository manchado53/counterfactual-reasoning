# Lab Notebook — Counterfactual Reasoning (CCE)

Read first. LOG = append-only (never edit). Rewrite only STATUS/NEXT.
Before /clear: append a dated LOG entry, especially DEAD ENDS.

## GOAL
ICLR 2027 (iclr.cc). Dates CONFIRMED 2026-08-13 from iclr.cc/Conferences/2027/Dates
and /CallForPapers (both agree; third-party trackers disagree — ignore them):
```
  ABSTRACT deadline   Sep 18 2026 AOE   <- the real gate: 36 days from 2026-08-13
  PAPER    deadline   Sep 25 2026 AOE      43 days
  reviews out         Nov 05 2026
  decisions           Dec 16 2026
  conference          Apr 26-30 2027 (location TBA)
```
The abstract deadline is a week before the paper deadline and is binding — no abstract,
no paper. So the planning horizon is 36 days, not 43. Old estimate "~late Sep" was close
on the paper date but hid the abstract gate.

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

## STATUS (2026-08-15)
Paper data = verified + frozen in paper/repro/ (master 861c0e3). NOT done experimenting — no
claim hits its target scenario count yet, and the paper's COVERAGE table is unchanged since
2026-06-02.

JaxNav (side branch) is now CLOSED as a negative result. Four budgets were run — 96k, 150k and
500k at 25 seeds/arm, the last with a uniform-replay control — and the promising 150k finding
("CCE prevents late collapse") did not survive the longer run. At 500k nothing separates the
arms significantly, every method degrades after ~ep 250k, and uniform replay collapses worst,
which rules out the mechanism story. Do not cite the 150k stability claim. All JaxNav data,
figures and videos are committed and rebuild from a clean checkout without the run tree.

36 days to the ICLR abstract deadline as of 2026-08-13. The paper still needs 2 more clean C2
scenarios and a C1 redo, and JaxNav did not become one of them.

## NEXT
0. **JaxNav REOPENED — rerun at `fill=0.3`, not 0.1.** Every JaxNav sweep ran on maps averaging
   0.91 interior obstacles, 36% of them completely empty (2026-08-19 LOG). The whole line may be
   a null because the map was near-empty, not because CCE fails. One number changes it. Driver
   `slurm/sweep_balance_ess.py` is ready; re-check the sanity gate (plain DQN must still solve it)
   before trusting any CCE comparison, and expect absolute win rates NOT comparable to fill=0.1.
   Also still untested at matched ESS: `consequence_aggregation='max'` — the flavour with
   P(>PER)=0.812 on the 500k data. This sweep used `weighted_mean`, which is 0.495 (a coin flip).
1. C4 Layer 1 — verify fixes are in code, then: does plain DQN beat the opponent at all?
2. Redo C1 on deterministic FL.
3. Fix 3 paper.tex numbers (FL-det → 25 seeds; FL-stoch → 0.67–0.75; SMAX PER → 0.71).
4. Recheck ICLR 2027 deadline.
5. [side track — REOPENED 2026-08-19, see item 0; the results below stand but were ALL
   measured at fill=0.1, i.e. on near-empty maps] JaxNav robotics-transfer, branch
   worktree-research+cce-robotics-transfer. Nothing outstanding to run; see the two 2026-08-15
   LOG entries for the result and the 2026-08-14 one for the claim it retracts.
   Summary of the whole line of work, so nobody reopens it by accident:
```
     budget   arms                          outcome
      96k     per/cce-max/cce-wmean         murky, nothing significant
     150k     same 3                        CCE+wmean 0/25 collapses vs PER 7/25  <- RETRACTED
     500k     + uniform control             uniform collapses MOST (14/25); CCE delays
                                            collapse, does not prevent it; nothing significant
```
   Everything needed to rebuild is committed: `docs/figures/real/claim2/jaxnav/data/`
   (manifests + per-seed curve npz for all four sweeps, 169+100 runs), 8 figures, 3 rollout
   videos, drivers in `slurm/`. Verified: a checkout of tracked files ONLY rebuilds every
   figure — `**/runs` and `**/experiments/` are gitignored, so this mirror is the only copy
   that survives a disk cleanup.

   IF IT IS EVER REOPENED, these are the traps already paid for:
   - Never read a sweep before every seed lands. Good agents finish the budget FIRST
     (short episodes), so an early read samples the winners — cost a 10pp overestimate once.
   - Never compare two runs at matched EPISODE COUNT if their epsilon schedules differ.
   - The tail-convergence check must be fitted per seed with a CI, not once to the pooled IQM
     curve; the pooled fit flips sign with the window.
   - "Same seed" does NOT reproduce a run here. Identical seeds/schedule reproduced only
     22/25, 18/25 and 13/25 seeds; the rest diverge from the first eval, most likely GPU float
     nondeterminism (CCE, which does the most GPU work, diverges most). Set deterministic XLA
     flags before any seed-level claim.
   - IQM: FIXED 2026-08-15. `jaxnav_holes_figures` now imports `compute_metrics.iqm`, which is
     rliable's aggregate_iqm, so there is one definition repo-wide. It previously trimmed one
     value per end and called that IQM. See the top LOG entry for which reported numbers moved
     (no conclusion did). `rliable` is now installed in the `counterfactual` env.
   - Post-hoc findings need fresh seeds. The 150k result was found by looking, confirmed
     nothing, and the 500k run reused the same seeds 0-24 by choice (continuation over
     independence), so it deepened the story without ever independently testing it.

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
- **JaxNav `fill` is NOT obstacle percentage.** `fill=0.1` on 8x8 gives 0.91 interior obstacles
  (2.5%) and 36% totally empty maps — the env default is 0.3 (4.56, 11% empty). An 8x8 map has 28
  border cells, so "28/64 occupied" is an EMPTY ROOM. Every sweep before 2026-08-19 used 0.1.
- **Multiplicative `mu_c`/`mu_delta` are exponents, not weights** — no cap at 1.0, and they
  multiply beta (`p_c^mu_c = (score+eps)^(beta*mu_c)`). mu_c=0 IS PER; mu_delta=0 is pure CCE.
- **The buffer's priority underflow fallback is SILENT** (`total==0 -> uniform`). A run can log
  the exponent it was asked for while actually sampling uniformly. Mixing is done in log space
  now; `ess_k_saturated` in `ess.jsonl` flags the degenerate case.
- **Concentration confounds every CCE-vs-PER comparison** unless matched: at a common exponent
  pure-CCE is ~2x sharper than pure-TD (ess_frac 0.47 vs 0.87). Use `target_ess_frac`.
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

### 2026-08-19 — **JaxNav ran on near-EMPTY maps all along** + ESS-matched replay comparison
Branch `experiment/jaxnav-ess-matched-mu`. Two findings; the first one re-contextualises every
JaxNav result in this notebook.

**1. THE `fill` GOTCHA (biggest thing here — check this before reading any JaxNav entry).**
Every JaxNav sweep ever run (11 files in `slurm/`) overrides the env default `fill=0.3` down to
`fill=0.1`, and this notebook describes that as "8x8, 10% obstacles". It is not 10%. Measured
over 200 sampled maps at the exact sweep config:
```
 fill   mean INTERIOR obstacles (6x6=36 cells)   % maps completely EMPTY
  0.1              1.00  (= 2.8%, not 10%)             33.3%    <- every sweep
  0.3              4.56                                10.0%    <- env default
  0.5              8.52                                 5.5%
 (11x11, fill 0.3: 11.50 / 3.9%.  11x11, fill 0.5: 19.55 / 2.5%)
 CONVERGED figures: 5 seeds x 2000 maps each. Earlier entries in this session
 quoted 0.91/36% and 1.08/31% from n=200 and n=300 probes -- same quantity,
 small-sample noise. Use the numbers above.
```
An 8x8 map has 28 border-wall cells; a "28/64 occupied" map is a bare empty room. The three maps
rendered for the rollout videos had 0, 1 and 0 interior obstacles.
CONSEQUENCE: the 2026-08-06 claim *"CCE wins with obstacles, loses without"* compared
`fill=0.0` (0 obstacles, always) against `fill=0.1` (0 obstacles, **36% of the time**). Treatment
and control overlap on a third of episodes — that contrast is much weaker than it reads.
WORSE: that pair also differed in MAP SIZE — easy was **6x6** fill=0.0, holes was **8x8** fill=0.1
(`slurm/sweep_easy.py` vs `sweep_holes_long.py`). So obstacles and map size moved together and
the obstacle contrast was ~1 obstacle. A clean redo needs fill=0.0 vs fill=0.3 at the SAME size.
NOTE `fill` was IDENTICAL (0.1) in every JaxNav run ever — 96k, 150k, 500k and the 2026-08-19
balance sweep — so it does NOT explain the 150k-good / 500k-bad reversal (that stays a budget
effect). It is a CEILING on the available signal, applied equally to every result, good and bad.
HYPOTHESIS (untested): JaxNav may not be a null for CCE, it may be a null because the map is
near-empty. In an open bordered room almost nothing is irreversible, so per-action return
distributions barely differ, TV ~ 0, and there is no signal to prioritise — the SAME estimator
failure already diagnosed on DoorKey (2026-08-06) and CVRP. Fix is one number: `fill=0.3`.
**Do not run another JaxNav experiment at fill=0.1.**

**2. ESS-MATCHED BALANCE SWEEP (40 runs, 40/40 COMPLETED, jobs 273145-273184).**
Motivation: every past CCE-vs-PER comparison changed TWO things — which signal ranks transitions,
AND how concentrated the sampler is. Measured at a common exponent: pure-TD ess_frac 0.87 vs
pure-CCE 0.47, i.e. CCE was ~2x sharper for free. A CCE win could always have been sharpness.
Built `ConsequenceReplayBuffer._solve_ess_exponents`: bisects the exponent scale so ess_frac hits
a target for ANY balance, recalibrated every 50 priority computations (~0 runtime cost: 3.18 vs
3.20 ms/call at 100k). New config keys `cce_balance` / `target_ess_frac` / `ess_recalib_every`;
realized ESS logged per eval to `ess.jsonl` so the matching is verifiable, not assumed.
Verified exact: ess_frac 0.600 +/- 0.02 on all 5 arms, 0 saturated evals.
```
 balance   AUC     P(>PER)      final IQM     <- 8 seeds, 250k, fill=0.1, weighted_mean
   0% PER  0.382     --           0.713
  25%      0.486    0.982         0.796
  50%      0.488    0.796         0.744
  75%      0.482    0.965         0.615
 100%      0.470    0.933         0.767
 TREND on AUC: Spearman rho +0.03, p 0.84  -> FLAT
```
Every CCE arm beats PER on AUC; the trend across balance is flat. So *any* CCE helps, *more* does
not — a step, not a dose-response. **Final win rate showed nothing (rho -0.17, p 0.31) because PER
catches up by ep 200k; C2 is a SPEED claim, so AUC/curve is the right readout, not the endpoint.**
NOT WRITTEN UP, because: (a) fill=0.1 above, and (b) PER's CI is huge [0.218,0.509] while CCE's
are tight — the gap is PER having bad seeds, which is exactly the "CCE's edge is stability" shape
RETRACTED on 2026-08-15 when it died at 500k. We are at 250k, inside the window where that dead
result was still alive.

**Also recorded:**
- **`P(CCE+max > PER) = 0.812` on the existing 500k data** — recovered from already-committed
  `curves_25seed_500k.npz`, never extracted because only two-tailed permutation tests were run
  (p=0.38). Both correct: 0.81 one-sided ~ p 0.19. This is the statistic Jeremy asked for on
  08-13 (said 80-90% reportable). CCE+wmean is 0.495 — a coin flip. So `max` is the live flavour
  and `wmean` (what this sweep used) is the flat one; `max` remains UNTESTED at matched ESS.
- Multiplicative `mu_c`/`mu_delta` are EXPONENTS, not weights — 1.0 is a default, not a cap, and
  `p_c^mu_c = (score+eps)^(beta*mu_c)`, so **mu_c multiplies beta**. mu_c=0 is exactly PER,
  mu_delta=0 is pure CCE, both 0 is uniform. The whole (1,1) corner was never moved off.
- Sharpening is NOT scale-free: at mu=(4,4) a dense score histogram gives ess_frac 0.50 but an
  FL-like 96%-zero histogram gives 0.04 (half the draws from 1.5% of the buffer). Never reuse an
  mu grid across envs — target an ESS instead.
- Underflow in the mixed priority is real but only past combined exponent ~40-60 at a 100k buffer
  (p ~ 1/N); the solver's bracket hit it, so `_mix_log` does the mixing in log space. The
  buffer's `total == 0 -> uniform` fallback is SILENT — `ess_k_saturated` now flags it.
- Colour: ColorBrewer Blues 5-class failed the normal-vision separation floor (adjacent dE 9.1 vs
  15 required) as overlapping lines. Figures use a validated 5-hue set (worst adjacent dE 18.8).


### 2026-08-15 (later still) — JaxNav figures now use the paper's IQM; some reported IQMs shift
`jaxnav_holes_figures.py` defined its own `iqm()` that trimmed ONE value from each end (23 of 25
seeds) and called the result IQM. IQM means the middle 50% — 25% trimmed per tail, 13 of 25 —
which is what `compute_metrics.py` computes via `rliable.metrics.aggregate_iqm`
(itself `scipy.stats.trim_mean(..., 0.25)`). Same word, two numbers.

Fixed by pointing the JaxNav module at the paper's definition: `compute_metrics.iqm()` is now
the single repo-wide IQM and `jaxnav_holes_figures` imports it. Repo-wide search confirmed the
trim-one version existed in that ONE file — every paper figure (FL, SMAX) already went through
`compute_metrics`, so **no paper number was ever affected**. `docs/_archive/mock_claim2_figures.py`
has a third, percentile-masking variant, but that tree is already marked do-not-cite.

`rliable` was missing from the `counterfactual` conda env (it lives in the broken `~/.local`
that `PYTHONNOUSERSITE=1` deliberately bypasses), so it is now installed into the env properly.
Only added `arch` + `rliable`; jax and jaxmarl verified still importable afterwards.

REPORTED IQM VALUES THAT CHANGE (means, std, all p-values and all collapse counts are
UNAFFECTED — they never used this function):
```
              old(trim-1)  true IQM
  96k   per      61.3%      62.7%     <- the true values match what the 2026-08-13 entry
        cce_max  70.1%      71.7%        already recorded (62.7/71.7/66.0), so the notebook
        cce_wm   63.2%      66.0%        was right and only the script had drifted
  150k  per      52.1%      56.4%
        cce_max  63.4%      64.9%
        cce_wm   64.7%      64.5%
  500k  uniform  38.7%      34.6%
        per      49.7%      53.2%
        cce_max  56.8%      64.1%
        cce_wm   49.3%      51.2%
```
The 2026-08-14 and 2026-08-15 entries below quote the old trim-1 IQMs; read the table above for
the corrected ones. **No conclusion changes** — every significance test and every collapse count
is computed from per-seed finals or per-seed curves, not from this aggregate. The IQM curve
panels in all 8 figures shift up slightly (500k peaks ~78% rather than ~74%) because the true
IQM trims the dead seeds out of the middle; shape, ordering and the decline after ep ~250k are
identical.

### 2026-08-15 (later) — 500k figures committed; the decline is visible in the IQM itself
Follow-up to the entry below, which closed with "no committed figure for this run yet". There
is one now: `fig_jaxnav_25seed_500k.png` and `fig_jaxnav_collapse_500k.png`, both in the house
two-panel format.

What the curve panel adds beyond the numbers:
- **The four arms are indistinguishable until ~ep 150k.** Whatever separation exists appears
  only in the back half of training, which is why the 96k and 150k runs each told a different
  story. Any JaxNav comparison stopped before ~200k is measuring noise.
- **Every arm peaks around ep 200-300k and then declines for the rest of training.** This is not
  a few seeds falling off a cliff dragging a mean down — the IQM, i.e. the middle of the
  distribution, turns over and falls. Uniform declines most steeply (to ~40%), CCE+max holds
  highest (~60%).
- **Every arm is bimodal at the end**: a healthy clump at 55-90% and a dead clump under 20%,
  with almost nothing between. So the reported means describe no actual seed, and that is the
  real reason the t-test and Mann-Whitney disagreed throughout this line of work.

Figure-code changes (`analysis/claim2/jaxnav_holes_figures.py`):
- `_power_arms` recognises `dqn-uniform` and returns only the arms a sweep actually contains;
  `fig_25seed_power` and `fig_collapse` iterate that list instead of a hardcoded triple, so one
  code path draws the 3-arm 96k/150k figures and the 4-arm 500k one.
- **Bug the 4th arm exposed**: the strip panel's x-limit was pinned to 3 arms, so CCE+wmean was
  drawn off-axes with its mean label floating outside the frame. The earlier 3-arm figures were
  unaffected.
- Dropped "(properly powered)" from the title. That power analysis was for a MEAN comparison at
  96k and was never redone for these budgets or for the variance/collapse endpoints; reasserting
  it on every rerun claimed more than had been checked.

Two verifications worth recording:
- The left panel of both 500k figures is the SAME curve (checked: identical seed sets,
  bit-identical IQM values, all 4 arms). They differ only in the right panel — final win rate
  vs drop from each seed's own peak. Final score alone cannot separate "climbed steadily to 55%"
  from "hit 85% and fell apart", which is why both views exist.
- Clean-checkout test repeated with the 500k data: a worktree containing only git-tracked files
  (no `runs/`, no `experiments/`) rebuilds all 8 figures, 25/25 coverage on all four arms.

### 2026-08-15 — JaxNav 500k + uniform control: the collapse result DOES NOT HOLD. Negative.
All 100 jobs COMPLETED, every seed at the full 500k (jobs 272275-272374; cce-max 272275-99,
cce-wmean 272300-24, per 272325-49, uniform 272350-74). This run kills the 2026-08-14 headline.

**Two things it establishes, both against us.**

1. **Collapse is a general DQN failure here, not something PER causes.** The uniform-replay arm
   collapses MORE than any other: 14/25 vs PER 9/25, CCE+max 7/25, CCE+wmean 11/25. So the
   mechanism story ("PER stops replaying mastered routes, so they are forgotten") cannot be the
   explanation — uniform replay never deprioritises anything and collapses worst of all.
2. **CCE does not prevent collapse. It delays it.** Truncating the SAME runs at increasing
   budgets shows the 150k result evaporating:
```
   collapse count /25      150k   250k   350k   500k
     uniform                 3      7     13     14
     per                     7      4      6      9
     cce_max                 1      4      4      7
     cce_wmean               1      8     10     11     <- was the 0/25 star at 150k
```
   At 150k this run reproduces the earlier one almost exactly (per 7, max 1, wmean 1 against the
   original 7/1/0) — so the metric and pipeline are sound and the old numbers were real. They
   just do not survive more training.

**Nothing is significant at 500k.** Final win rate: uniform 39.3%, PER 49.3%, CCE+wmean 49.2%,
CCE+max 56.2%. Permutation tests: CCE+max vs PER +6.8pp p=0.38; CCE+wmean vs PER -0.1pp p=0.99;
CCE+max vs uniform +16.9pp p=0.0498 (and that is 1 of 5 tests, so it does not survive
correction); PER vs uniform +10.0pp p=0.23. The spread result that was the whole 2026-08-14
story is gone: Brown-Forsythe PER vs CCE+wmean was p=0.0005 at 150k, now p=0.25 (std 27.3 vs
33.3pp — CCE+wmean is now the MORE variable arm, not the less).

**So the 2026-08-14 entry's conclusion is retracted.** "CCE's edge is stability" was true at 150k
and false at 500k. It was a post-hoc finding, flagged as needing confirmation, and the
confirmation went against it. Recording this rather than quietly dropping it: the honest summary
of JaxNav is now that CCE+max is nominally best on both final score and collapse count, no
comparison reaches significance, and every arm degrades badly with long training.

**Reproducibility caveat found here.** Seeds and epsilon schedule were identical to the 150k run,
so its first 150k episodes should have been bit-identical. Only 22/25 (per), 18/25 (cce_max) and
13/25 (cce_wmean) seeds actually matched; the rest diverge from the FIRST eval. numpy is seeded
in all four trainers (`np.random.seed(config['seed'])`), so the likely cause is GPU float
nondeterminism (XLA reductions), consistent with CCE — which does the most GPU work — diverging
most. Not proven. Practical consequence: "same seed" does NOT guarantee a reproducible run here,
so continuation runs are only partly continuations, and any future claim resting on seed-level
comparison needs `XLA_FLAGS=--xla_gpu_deterministic_ops=true` or similar first.

Data committed: `docs/figures/real/claim2/jaxnav/data/manifest_25seed_500k.json` +
`curves_25seed_500k.npz` (100/100 runs). Driver `slurm/sweep_holes_25seed_500k.py` (has
--dry-run). Timing came in near projection: uniform 2.15h, PER 3.29h, CCE ~9.6-10.0h mean
(max 12.1h) against a 20h limit.

STILL OWED: the figure module only knows 3 arms (`_power_arms`), so there is no committed figure
for this run yet — the numbers above are from direct analysis. Adding a uniform arm to the figure
code is the next task.

### 2026-08-14 — JaxNav 150k/25-seed DONE: CCE's edge is STABILITY, not a higher ceiling
All 75 jobs COMPLETED, zero failures, all 25 seeds/arm at full budget. This is the cleanest
JaxNav result so far and it reframes what CCE is doing on this task.

**The arms reach the same ceiling and differ at the floor.**
```
150k, 25 seeds/arm, holes map          mean    IQM    std    worst5   best5
  PER                                  51.5%  52.1%  25.5pp   12.8%   79.5%
  CCE+max   (old buggy aggregation)    62.5%  63.4%  14.1pp   40.5%   78.4%
  CCE+wmean (the issue-#3 fix)         64.6%  64.7%   8.8pp   52.5%   76.4%
```
best5 is the same for all three (76-80%). The whole difference is the bottom of the
distribution. Brown-Forsythe on the spread: CCE+max p=0.0119, CCE+wmean p=0.0005.
Bootstrap 95% CI on the mean gap vs PER: max +11.0pp [0.0,+22.2], wmean +13.1pp [+2.9,+23.8].

**Why: PER learns and then throws it away.** 7/25 PER seeds end >25pp below their own smoothed
peak (e.g. job 272176 peaked 75.6% at ep 131k, finished 14.6%). CCE+max 1/25, CCE+wmean 0/25.
Fisher p=0.0488 / p=0.0096. See `fig_jaxnav_collapse_150k.png` — the failure is invisible in an
IQM curve and obvious per seed.

**This is why 96k showed nothing.** The PER collapses start at ep 104k-134k, past the old
budget. At 96k the collapse counts are PER 2/25, max 1/25, wmean 1/25 (p=1.0) and the spreads
are identical (~16pp, BF p=0.74/0.99). Nothing was there to find.

**t-test and Mann-Whitney disagree on purpose** (wmean vs PER: t p=0.021, MW p=0.214). Correct
behaviour, not a bug: the medians are close (60.8% vs 64.8%), so ranks barely move; the means
separate because PER has a failure tail. Report the spread and the collapse count, not the
mean gap — the mean gap is a side effect.

**The aggregation fix now looks right.** At 96k the buggy `max` scored highest, which muddied
the issue-#3 story. At 150k with everything converged, `weighted_mean` is best on every
reliability measure (std 8.8pp vs 14.1pp, 0 collapses vs 1, worst5 52.5% vs 40.5%), while the
two are tied on the mean (-2.1pp, p=0.53). So the fix helps where it matters.

CAVEAT: the 96k and 150k sweeps differ in epsilon decay (40000 vs 62500) as well as length, so
cross-run comparisons are confounded. Every claim above is a WITHIN-150k-run comparison —
same schedule, same seeds — except the "why 96k showed nothing" timing note.

Method fixes this session (all in `analysis/claim2/jaxnav_holes_figures.py`):
- coverage printed per arm; seeds short of 95% of the manifest budget are excluded, not
  silently averaged in at whatever episode they died on.
- convergence check refitted PER SEED with a CI across seeds. The old last-minus-first version
  is endpoint-noise dominated and flips sign with the window (PER: -1.2pp at 3k, +15.9pp at
  20k on identical data). See the correction in NEXT — the 2026-08-13 entry's "PER flat, CCE
  rising/falling" read is an artifact and should not be cited.
- collapse detection compares smoothed peak to smoothed final (both 41 evals). Comparing a
  smoothed peak to a raw 5-eval tail flags noise as collapse; that mismatch was inventing a
  CCE+wmean "collapse" whose curve is visibly flat to the end.
- the full threshold x smoothing grid is printed rather than one hand-picked cell, because
  the 25pp/41-eval choice was made after seeing the data. Direction is robust: PER is strictly
  highest in 20/20 cells at 150k (and only 5/20 at 96k, i.e. no effect there).

Two measurement traps that produced WRONG numbers before being caught — both in NEXT:
never read a JaxNav sweep before every seed lands (fast seeds are the good ones, Spearman
+0.49; cost a 10pp overestimate), and never compare runs at matched episode count when the
epsilon schedules differ (made healthy CCE arms look catastrophic).

**IS ANY OF IT SIGNIFICANT? One thing, and it is not the thing we set out to test.**
The t-test and Mann-Whitney disagreed on the mean (p=0.021 vs 0.214) because PER's dead tail
flatters the t-test, so the arbiter is a permutation test (200k shuffles, no normality
assumption). With Bonferroni over the 6 tests run:
```
                                        raw p    x6      survives
  wmean vs PER  mean (permutation)      0.0183  0.110    no
  max   vs PER  mean (permutation)      0.0639  0.384    no
  wmean vs PER  SPREAD (Brown-Forsythe) 0.0005  0.0033   YES
  max   vs PER  SPREAD                  0.0119  0.071    marginal
  wmean vs PER  collapse count (Fisher) 0.0096  0.058    marginal
  max   vs PER  collapse count (Fisher) 0.0488  0.293    no
```
So: the planned claim ("CCE speeds learning") does NOT reach significance here. The spread
result does. BUT it is POST-HOC — found by looking at the data, with the 25pp/41-eval choice
made after the fact, and it did not appear at 96k. Treat it as a hypothesis needing a
pre-registered confirmation, not a result. Do not write it up as confirmed.

**Mechanism of the collapse (checked, not assumed).** Collapsed seeds' episodes get LONGER as
win rate falls (e.g. 97 -> 151 steps, cap 200) while healthy seeds get shorter (157 -> 90). So
the robot stops reaching the goal and wanders to timeout — forgetting, not extra crashing.
Plausible cause, NOT proven: PER prioritises by TD error, so a route that has been mastered
stops being surprising, stops being replayed, and is forgotten; CCE prioritises by whether the
action choice changed the outcome, which stays high at decision-critical states even once they
are predicted well. UNTESTED because the sweep has no uniform-replay arm — cannot separate "PER
causes it" from "PER fails to prevent a general DQN instability" (vanilla DQN already collapsed
on JaxNav once, see 2026-08-06). One 25-seed uniform run at 150k settles it; uniform runs at
PER speed (~1h), and `--algorithm dqn-uniform` already exists.

**Ruled out as causes** (all checked): jobs all COMPLETED and ran the full 150k with 600 evals;
fall is gradual over 16k-46k episodes, not an instant cliff; collapsed seeds spread over 5 nodes
with healthy seeds on the same nodes; zero NaN/inf in all 75 runs; log warnings identical in
healthy and collapsed runs; collapsed seed sets do not overlap between arms (PER 0,6,7,8,10,14,20
vs cce_max only 2 vs cce_wmean none), so it is not a cursed-seed artifact.

**CAVEAT on the word "IQM" in these figures.** `jaxnav_holes_figures.iqm()` trims ONE value from
each end (n=25 -> 4% trim), which is NOT the interquartile mean the paper's `compute_metrics.py`
uses via rliable (middle 50%). They differ most for PER, exactly because of its failure tail:
PER 52.1% (this script) vs 56.4% (true IQM). Fix or rename before any of these numbers go near
the paper.

**Setup, for the record.** Every episode draws a NEW random map, start and goal together
(`reset(key)` -> `sample_test_case`), 8x8, fill=0.1, goal_radius=0.8, max_steps=200, 15 discrete
actions, 205-dim obs. Training runs 256 envs in parallel with auto-reset to a fresh map. Eval =
100 fresh random maps every 250 episodes, and the eval maps CHANGE every evaluation — so the
curve carries map-draw luck on top of counting noise, which is why 41-eval smoothing is needed.
Worth fixing a held-out eval map set in any follow-up run; it would sharpen every curve here.

Jobs (Rosie): 272116-272140 cce-max, 272141-272165 cce-wmean, 272166-272190 per.
REPRODUCIBILITY: `**/runs` and `**/experiments/` are gitignored, so the raw tree is NOT in git.
Everything needed to rebuild the figures IS committed under
`docs/figures/real/claim2/jaxnav/data/` — `manifest_25seed_{power,150k}.json` plus
`curves_25seed_{power,150k}.npz` (per-seed episode/win-rate arrays, float64). The figure module
falls back to that cache automatically when the run tree is missing; verified to reproduce every
statistic bit-exactly with `RUNS` pointed at a nonexistent path. Regenerate the cache after a new
sweep with `export_cache(manifest_path, tag)`.
Figures: `fig_jaxnav_25seed_150k.png`, `fig_jaxnav_collapse_{96k,150k}.png`. Rerun with
`PYTHONPATH=<worktree>/src python -m counterfactual_rl.analysis.claim2.jaxnav_holes_figures`.
Videos of the best seed per arm on identical maps: `docs/figures/real/claim2/jaxnav/video/`,
regenerate with `...analysis.claim2.jaxnav_rollout_video <outdir>` (needs last.pkl, i.e. the run
tree — weights are too big to commit).

OPEN: is "CCE prevents late collapse" a paper claim or a side note? It is a different claim
from C2-as-written ("speeds learning") — this is "doesn't fall over". Needs a decision, and
if it counts, it needs replication on a paper env (FL-det long-run would be the cheap test).

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
