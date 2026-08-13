# CCE suitability predictor — plan

## Purpose (read this first)
These metrics are **decision tools for us**, not (yet) paper material. The job is to *understand
what makes CCE work*, so we can decide which environments / directions to **pursue or avoid**
before burning compute. They only need to be directionally useful for steering our own effort.
If a clean result later emerges (esp. the slip sweep), we can promote it to the paper — but the
heavy "publishable predictor" machinery (pre-registration, LOEO cross-val, calibrated regression)
is **optional, only-if-we-publish**, not required for the internal tool.

## What we're doing
Build a small set of **cheap, rollout-based metrics** that tell us, for any environment,
*whether CCE (our consequence-based replay priority) will actually help* — and **why** —
without paying for a full training run. Measure a new env (Connect Four, SMAX, anything) →
get a rough read on win / no-gain → decide whether it's worth chasing.

## Why
We have one clean win (FrozenLake deterministic) and a pile of nulls/marginals. We don't yet
understand *what property of an environment* makes CCE work. Understanding that — even roughly —
is more valuable than one more env result: it tells us where to point our remaining time.

## The organizing idea (after literature review + adversarial review)
Don't invent 5 ad-hoc numbers. Frame everything under the known normative theory of replay
(Mattar & Daw 2018):

    worth replaying  =  GAIN  ×  NEED
    CCE estimates the GAIN half (how much the action changed the outcome).
    Our metrics are the CONDITIONS under which a GAIN-estimator helps.

## The metric set (revised after review — this is what we lock)
GAIN is real & visible:
- **SNR** — between-action vs within-action return variance. **Roll out GREEDY after the
  forced action** so the denominator is environment noise only (not our exploration noise).
  Lead forecaster. (≈ noise-normalized action gap; Bellemare 2016.)
- **Concentration** — visit-weighted top-k stakes mass `Σ_topk C(s)d(s) / Σ C(s)d(s)`.
  (Gini only as a secondary descriptor — raw Gini over visited states is gameable by junk states.)

Does it beat PER (not just uniform):
- **DISTINCT-TD** = `1 − |Spearman(CCE-priority, |TD|)|`. **The key metric.** If CCE ranks like
  TD-error, it can't beat PER. This was missing and is the one that predicts our headline.

Is the gain reachable / revisited:
- **NEED** — forward-looking discounted occupancy of high-stakes states (successor representation
  `(I−γT)^-1`; tabular exact in FL, visitation-EMA in deep envs). Replaces the hacky overlap.
- **HORIZON-FIT** = `cf_horizon / effective_horizon(env)` (Laidlaw 2023). <1 → rollout estimator
  truncates the signal → expect CCE to underperform.

Truth + sampler sanity:
- **GAIN-FIDELITY** = Spearman(CCE-priority, **exact Q*-spread / EVB**) — only vs ground truth,
  never vs C(s) itself (that comparison is circular). FrozenLake only.
- **ESS / precision@k** — on the realized replay distribution. ESS is a *collapse alarm only*
  (low ESS ≠ bad; CCE is supposed to be spiky). Track precision@k / recall@k of drilled-vs-stakes.

Dropped: standalone Gini-as-forecast, NEED-OVERLAP double-threshold, circular Score-vs-C(s).

## Known pitfalls the review surfaced (must respect)
- **Policy dependence**: every metric depends on which π we roll out. Pre-register one fixed cheap
  policy (random + short warmup), and report a sensitivity check (random / 10% / 30% warmup).
- **Snapshot vs trajectory**: "CCE helps" is an integral over a moving state distribution. Compute
  metrics at multiple training snapshots, not one. Track DRIFT of C(s) if needed.
- **Cost honesty**: report forecast-compute / training-compute ratio. NEED needs a competent
  policy to be meaningful → quantify, don't claim "free."
- **Deep-RL transfer**: tabular hardness metrics predict deep-RL poorly (Pharos 2025). Must show
  the metrics predict in SMAX (deep), not only tabular FrozenLake.

## The outcome variable (what the metrics must predict)
Standardized, paired, vs the *stronger* baseline:

    benefit_e = ( mean_s AUC_CCE  −  max(mean AUC_PER, mean AUC_uniform) ) / pooled_SD
    AUC = normalized area under the eval-success curve (speed, not just ceiling)
    same seed list across CCE/PER/uniform; paired bootstrap (BCa) 95% CI over seeds.

## The real evidence — controlled sweeps, NOT a 4-env map
4 envs × 3 metrics = a line through 3 dots = overfit. The credible evidence is a within-env
dose-response curve where one knob moves and everything else is held fixed:

1. **SLIP SWEEP (experiment #1, run first).** FrozenLake, fixed 8×8 map, slip p ∈
   {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33}. Predict: SNR falls monotonically AND benefit_e
   falls with it. Confirms "noise kills CCE" on the exact axis our det-win / stoch-null anchors
   straddle. Falsifies the whole thesis cheaply if flat/non-monotone.
2. **Concentration sweep.** Synthetic gridworld, tunable # of pivotal states, fixed difficulty
   and path length. Predict benefit_e ↓ as stakes spread out.
3. **NEED sweep.** Same stakes, move pivotal states on-path vs off-path. Predict benefit_e ↓ when
   high-stakes states are rarely visited. (cleanest dissociation test.)

Plus the FrozenLake **bridge**: Spearman(C(s) from rollouts, exact Q*-spread) confirms the cheap
estimate is honest before we trust it on Connect Four / SMAX.

## Build order
1. One function: rollout-sweep over an env+policy → the pile (per-state per-action returns + visits).
2. From the pile compute SNR(greedy), Concentration, NEED, DISTINCT-TD (needs |TD| from a net),
   GAIN-FIDELITY (FL only), horizon-fit.
3. Run the FrozenLake bridge (cheap ≈ exact). Then the SLIP SWEEP (experiment #1).
4. Only after the sweep curves hold: draw the cross-env map as *exploratory*, with LOEO CV.

## Status
Plan stage. Branch: research/cce-buffer-diagnosis. Mock figures in docs/figures/mock_preview/.
Nothing trained yet for this line of work.
