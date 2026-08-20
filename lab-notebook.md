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
CVRP        WIN ✓ ρ .52→.67→.65        NULL — CLOSED, ~2,000 runs
(logistics)  exact oracle, 3 seeds      (domain has no dead decisions)
```
CVRP Claim 2 is CLOSED as a negative result. Cause measured: CCE is a rare-event detector and
routing has ~24% zero-stakes states vs FrozenLake's 86%. Survived every knob (aggregation,
budget, capacity, size, cf_gamma, 4 CCE hyperparameters). Do not re-open without a NEW mechanism.

## STATUS (2026-08-19)
Data so far = verified + frozen in paper/repro/ (master 861c0e3).
NOT done experimenting — no claim hits its target scenario count yet.

Two things landed 2026-08-18/19 on branch research/cce-cvrp-logistics:
1. **CVRP Claim-2 is CLOSED (negative).** ~2,000 runs, 6 SLURM arrays, 0 failures, best CCE cell
   P(beats PER)=0.594 while random replay reached 0.611. The pre-registered inverted-U was
   FALSIFIED. Keep CVRP as a Claim-1-only environment.
2. **LABELLING BUG, needs a human decision.** `consequence_aggregation='weighted_mean'` silently
   executes as `max()` in `frozen_lake` and `cvrp` (they never pass `action_probs`; chess and
   connect_four do). The FL results are still SOUND — max there is binary but selective, zero at
   86.3% of states — so this is a paper-wording fix, not a re-run. Code behaviour left UNCHANGED
   so paper/repro/ still reproduces; the fallback now emits a RuntimeWarning.

**The strongest thing we learned is transferable:** CCE's prerequisite is a BIMODAL stakes
distribution — decisions that are either free or critical, with little in between. Measured as
middle-band mass of the normalised exact stakes: FL-det 0.0% (WIN), FL-stoch 62.3% (null),
routing 68-75% (null). NOTE: an earlier version of this line said "high fraction of zero-stakes
states"; that was the CCE score's zero fraction, not the oracle's, and it is superseded — see the
2026-08-19 CORRECTION entry. Still fitted post hoc on 4 points; needs a prospective call.

## NEXT
1. **DECIDE: paper wording for aggregation.** It ran `max`, the paper says weighted mean. Relabel
   (recommended — the FL probe shows max is the selective, correct-behaving choice there), or
   re-run FL with true averaging. Do NOT silently change `analysis/metrics.py`.
2. **Pick the next C2 env by DEAD-STATE FRACTION, not by domain appeal.** Measure it from rollouts
   first (suitability pipeline); target >50% zero-stakes states. Routing failed because it is ~24%.
3. C4 Layer 1 — verify fixes are in code, then: does plain DQN beat the opponent at all?
   (C4 is a plausible high-dead-state env: most board positions are not pivotal.)
4. Redo C1 on deterministic FL.
5. Fix 3 paper.tex numbers (FL-det → 25 seeds; FL-stoch → 0.67–0.75; SMAX PER → 0.71).
6. Recheck ICLR 2027 deadline.

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

### 2026-08-20 — OPTION A built. **GATE 1 PASSED (stranding, not windows). GATE 2 FAILED at 0% solved.**
Added `time_windows` and `allow_stranding` to `envs/routing_budget.py`, both defaulting OFF so
every committed budget-mode result reproduces (asserted in tests). `spent` is reinterpreted as
TIME, matching the standard VRPTW formulation. The exact oracle needed NO change — every leg
still advances the clock, so the DAG backward pass over spend still solves it. 29 tests: the DP
matches brute force with windows on, with stranding on, and with both.

**GATE 1 (exact oracle, no training).** Share of decision states by how much the choice matters:
```
                             barely   MIDDLE    crit
FrozenLake det (CCE wins)     50.9%     0.0%   49.1%
baseline (today)              11.5%    75.1%   13.4%
windows x3                    17.8%    82.1%    0.1%   <- WORSE
windows all                   14.7%    84.8%    0.5%   <- WORSE
strand, terminal              14.1%     9.6%   76.3%   <- collapses the middle band
windows x3 + strand term      18.8%    15.5%   65.7%
```
**TIME WINDOWS MAKE IT WORSE ALONE — my prediction was wrong.** I argued for them from SVRPBench,
where they are the headline difficulty lever (+536-648% cost, learners to 85-88% feasibility). On
this instance they OVER-DETERMINE the route: one viable ordering survives, so nothing is a real
choice and critical states fall to ~0.1%. Kept as a flag, defaulted off. **Stranding is the whole
lever**: 75.1% -> 9.6% middle band, purely by making catastrophe reachable.

CAVEAT, recorded not glossed: the new shape is skewed the OPPOSITE way from FrozenLake — 76.3%
critical vs 49.1%. A budget sweep shows the trade is monotone with no balanced setting available:
```
mult  barely MIDDLE  crit   fail-states
0.80   14.1%   9.6% 76.3%   38.8%
0.95   13.7%  13.6% 72.7%   22.6%
1.10   14.8%  24.8% 60.4%   12.1%
1.25   20.2%  39.8% 40.1%    5.8%
1.60   75.5%  13.7% 10.8%    0.6%
```

**GATE 2 (DQN-uniform only, 4 seeds, 2000 eps) — FAILED. Target 30-60% of seeds solved; got 0%.**
```
config                    final mean  solved  ended-at-zero  best seen
strand+stepwise B=0.95x       0.444    0/4        2/4          0.889
strand+terminal B=0.80x       0.531    0/4        1/4          0.844
strand+terminal B=0.95x       0.583    0/4        1/4          0.889
```
Nothing ever reaches the optimum. A seed-0 curve reads `0.00 0.00 0.00 0.78 0.00 0.00 0.00 0.00
0.89` — the greedy policy oscillates between decent and TOTAL LOSS, because one bad Q-estimate
drives the truck somewhere it cannot return from. Outcomes are genuinely bimodal now (seeds do
end at exactly 0, which never happened before), but the task overshot the useful window: it went
from "everyone solves it" straight to "nobody does".

**CALIBRATION (42 baseline runs, 6000 episodes).** Solve rate = seeds whose final greedy policy
matches the exact oracle:
```
budget   solved        zeros   converged?
0.95x    1/6   17%     1/6     no - still climbing (+0.019 over the last third)
1.10x    0/6    0%     1/6     yes
1.14x    2/8   25%     2/8     yes
1.17x    1/8   12%     0/8     no - still rising (+0.012)
1.21x    1/8   12%     1/8     yes
1.25x    6/6  100%     0/6     yes - solved by episode 1500
```
**ROUTING DIFFICULTY IS A STEP FUNCTION, NOT A DIAL.** Solve rate runs 12% -> 25% -> 100% across
FOUR budget units (B=42 -> 44). Nothing sits in the 30-60% band that makes a comparison
informative. This is the same fact the stakes distribution reported, in a second currency: the
budget sweep there was equally monotone with no balanced setting.

Note the non-monotonicity (1.10x scores WORSE than 0.95x): "solved" means matching the oracle
exactly, and the oracle target MOVES with the budget - 9 customers at 0.95x, all 10 at 1.10x.
Serving 10 is harder than serving 9 even with more slack.

**IN PROGRESS — SLURM 273792, 30 runs, 25,000 episodes** (4x longer) at 0.95x / 1.14x / 1.17x,
the three budgets that had NOT converged. Tests whether the sub-30% solve rates are a
training-length artefact rather than a property of the environment. **Still no CCE arm in this
configuration**, deliberately: the budget is calibrated on the baseline alone, so the environment
cannot be tuned to favour our own method.

### 2026-08-19 — **CORRECTION + SHARPER PREDICTOR: it is BIMODALITY, not the dead-state fraction.**
My earlier entry today framed the thesis as "CCE needs a high fraction of ZERO-stakes states
(FL 86% vs routing 24%)". Those numbers came from the CCE SCORE's zero fraction, not from the
exact oracle, and the oracle does not say that. Measured properly — exact Q* spread per decision
state, normalised by the per-env max:
```
                       barely     MIDDLE    critical        C2 result
                        <5%        5-50%      >50%
FrozenLake det         50.9%        0.0%     49.1%          WIN
FrozenLake slippery    24.5%       62.3%     13.2%          null
Budget routing 0.70x   10.9%       75.2%     13.9%          null
Budget routing 1.00x   18.3%       68.1%     13.6%          null
```
**FL-deterministic — the ONLY env CCE wins on — has EXACTLY ZERO states in the middle band.**
It is perfectly bimodal: every decision is either free or critical, nothing partly matters.
Both nulls have a fat middle (62%, 75%). Routing is NOT short of critical states (13.9%, same
order as FL-slippery); it is short of SEPARATION.

**Mechanism.** CCE ranks states and replays the top ones. That isolates something only if there
is a clump to isolate. With a smooth spread, "top 10%" is an arbitrary cut through a ramp — the
11th percentile is barely different from the 9th — so the priority ordering carries little
information beyond noise. With a bimodal spread, top-k lands exactly on the critical cluster.

**REVISED PREDICTOR (supersedes the dead-state-fraction version above):** middle-band mass of the
normalised oracle stakes. Low middle -> CCE can win. Fat middle -> it cannot. Separates all four
of our data points cleanly, which the dead-state fraction did not.

**CAVEAT — this is 4 data points and it was fitted AFTER the fact.** It retro-predicts; it has not
predicted anything yet. The honest next step is a PROSPECTIVE test: measure middle-band mass on a
candidate env BEFORE running C2, write the prediction down, then run. Connect Four is the obvious
candidate. Do not put this in the paper as a law until it has called one in advance.

### 2026-08-19 — 12-customer sweep agrees: null. All six arrays complete, ~2,000 runs, 0 failures.
`ring12mean` (360 runs, 12 customers, mean aggregation) finished 360/360. Best CCE cell
P(beats PER)=0.514; most cells 0.17-0.46. The larger instance reproduces the 10-stop conclusion
exactly, so the null is not an artefact of instance size. Claim-2 in routing is closed.

### 2026-08-19 — **DECISIVE: corrected sweep (720 runs) is a CLEAN NULL. Registered prediction FALSIFIED.**
The mean-aggregation + cf_gamma=1.0 sweep finished 720/720 with zero failures: 4 budgets x 3
capacities x 5 arms x 12 seeds, every cell complete. P(beats PER) on AUC:
```
capacity 5    0.70x  0.80x  0.90x  1.00x    capacity 10   0.70x  0.80x  0.90x  1.00x
CCE+TD add    0.587  0.521  0.594  0.438    CCE+TD add    0.324  0.509  0.552  0.321
CCE+TD mul    0.369  0.376  0.463  0.312    CCE+TD mul    0.526  0.508  0.562  0.209
CCE-only      0.298  0.334  0.349  0.201    CCE-only      0.478  0.471  0.511  0.238
DQN-Uniform   0.477  0.302  0.415  0.402    DQN-Uniform   0.373  0.502  0.611  0.226
```
Best CCE cell anywhere = 0.594. **DQN-Uniform reaches 0.611** (cap10/0.90x), out-scoring every CCE
arm in that cell — random replay beating PER is the same noise signature that discredited the
original CVRP null, and it recurs here. **NULL.**

**THE PRE-REGISTERED INVERTED U IS FALSIFIED.** No consistent mid-dial peak; the weak bump moves
between capacities (cap5 flat-then-drop, cap10 peaks at 0.90x, cap6 monotone-ish). We predicted
it in advance, tested it on two independent axes, and it did not hold. Recording that as a
falsification rather than reinterpreting the shape after the fact.

**FULL LIST OF WHAT WAS TRIED, ALL NULL.** aggregation {max, mean} x budget {0.60..1.30} x
capacity {10, 6, 5} x instance {10-stop, 12-stop} x cf_gamma {0.99, 1.0} x CCE knobs
{cf_n_rollouts 20/60, mu 0.25/0.5/1.0, score_interval 5/20, additive/multiplicative}.
Total ~2,000 runs across 6 SLURM arrays, 0 failures.

**VERDICT: routing is a NEGATIVE result for Claim 2 and should be reported as one.** The cause is
now measured, not guessed: CCE is a rare-event detector and routing has ~24% zero-stakes states
against FrozenLake's 86%. Every knob we turned changed the SHAPE of the score; none could create
dead decisions that the domain does not contain.

**KEEP CVRP AS A CLAIM-1 ENVIRONMENT ONLY** (rho 0.52 -> 0.67 with an exact oracle stands), and
use the Claim-2 null as the evidence for the suitability thesis — the dead-state fraction is a
cheap, rollout-measurable predictor that retro-predicts FL-det (86%, WIN), FL-stoch (NULL) and
routing (24%, NULL). That is a stronger contribution than a fourth win would have been, and it is
falsifiable: find a domain with a high dead-state fraction and CCE should win there.

### 2026-08-19 — **FrozenLake is NOT broken by the aggregation bug — and this explains everything.**
Measured FL's CCE score in the live training loop under both aggregations (1500 eps, 8x8):
```
                     distinct   at ceiling   ZERO     gini
FL determ.  max          2         13.7%     86.3%    0.000    <- what the paper ran
FL determ.  mean         4          3.6%     92.7%    0.177
FL slippery max         18          0.0%     55.6%    0.386
FL slippery mean        41          0.0%     52.5%    0.434
budget routing max       2         76.5%     23.5%    0.000    <- for contrast
```
**PARTIAL WALK-BACK of the alarm I raised.** max() IS binary in FL-deterministic — but it is
binary and **SELECTIVE**: it stays at ZERO for 86.3% of states. That is precisely the rare-event
detector CCE is supposed to be, and it is why FL-det is the headline win. In FL-slippery max is
not even binary (18 distinct values) because stochastic rollouts spread the return distributions.
So the published FL numbers rest on a sound mechanism; the defect is one of LABELLING (the config
and paper say weighted_mean, the code ran max), not of validity. Fixing the label is cheap;
re-running FL is not required by this evidence.

**THE REAL REASON ROUTING KEEPS FAILING — and it is not the metric.** The same max() score is
selective in FrozenLake (86% zeros) and saturated in routing (24% zeros). The metric did not
change; the DOMAIN did.
```
                     states where the action choice does NOT matter
FrozenLake 8x8 det                  86.3%      <- CCE wins here
budget routing (any setting)        ~24%       <- CCE cannot win here
```
This is the SAME number the plan file measured on day one from the exact oracle — routing had
0.7-3% "dead" states vs FrozenLake's 50.9% — and it has now survived EVERY intervention we
threw at it: capacity dial, budget dial, aggregation fix, cf_gamma fix, instance size, and four
CCE hyperparameters. **Routing has no dead zones. In a grid you can stand on open ice where every
move is equivalent; in routing every choice picks among spatially spread stops, so every choice
matters a little and almost none matter a lot.** CCE is a rare-event detector, and routing has no
rare events.

**SHARPENED THESIS (the transferable contribution).** CCE helps when a large fraction of decisions
are INCONSEQUENTIAL, so that a cheap score can find the few that are not. Its prerequisite is not
determinism, not discrete reward, not stakes-concentration alone — it is a high proportion of
zero-stakes states. This is measurable from rollouts BEFORE spending compute (it is the
suitability pipeline's "dead state" fraction), and it retro-predicts every result we have:
FL-det (86% dead, WIN), FL-stoch (graded but noisy, NULL), CVRP/budget routing (~24% dead, NULL).

### 2026-08-19 — MAX-aggregation control COMPLETE (936 runs): another NULL. Corrected sweep running.
The four sweeps launched before the aggregation bug was found are finished and clean (600 main +
240 cap5 + 96 tuning, 0 failures; ring12-max cancelled at 76 runs to free nodes). They measure the
BINARY max() score, so they are the CONTROL, not the test. Verdict: **no reliable CCE win.**
P(beats PER) on AUC, 12 seeds per arm, complete cells only:
```
capacity 10   0.60x  0.70x  0.80x  0.90x  1.00x     capacity 5   0.70x  0.80x  0.90x  1.00x
CCE+TD add    0.535  0.416  0.638  0.559  0.328     CCE+TD add   0.631  0.539  0.479  0.506
CCE+TD mul    0.496  0.382  0.467  0.500  0.210     CCE+TD mul   0.375  0.448  0.425  0.312
CCE-only      0.465  0.326  0.499  0.414  0.266     CCE-only     0.375  0.459  0.230  0.278
DQN-Uniform   0.389  0.373  0.502  0.611  0.226     DQN-Uniform  0.477  0.302  0.415  0.402
```
CCE+TD(add) is the only arm that ever leads (0.631 / 0.638) and it never reaches a convincing
margin. **DQN-Uniform hits 0.611 at cap10/0.90x** — random replay "beating" PER again, the same
noise signature that discredited the original CVRP null. The registered inverted-U is at best
weakly visible in cceadd and does not replicate across capacities. Treat as NULL.
Tuning knobs did not rescue it either (cap6/0.70x): r60 0.366, mu50 0.459, mul-r60 0.465,
score_interval=5 0.279 — **fresher scores were WORSE**.
Figures: `docs/figures/real/claim2/fig_c2_cvrp_budget_dial.png` + `cvrp_budget_sweep_summary.json`.

**This control is exactly what the aggregation bug predicts.** A binary score that fires at ~76% of
states cannot prioritise, so CCE degenerates toward PER-with-noise. The real test is the
mean-aggregation sweep (273435, 720 runs) plus ring12-mean (273492, 360 runs), both running.

### 2026-08-19 — **BUG: `weighted_mean` has been running as `max`.** CCE's score was BINARY.
Probing the LIVE training loop (not offline — the notebook already records one offline probe
lying about this) showed the CCE score in budget mode was still binary: **2 distinct values,
~76% of scored states at the ceiling**, gini 0.000. Budget mode was built to fix precisely that,
so the premise looked dead. It was not the environment.

**ROOT CAUSE.** `analysis/metrics.py::compute_consequence_metric` honours `'weighted_mean'` only
when `action_probs` is passed. Without it, control falls through to `return max(...)`.
```
passes action_probs:  chess YES   connect_four YES
does NOT pass:        frozen_lake NO   cvrp NO      <- config says weighted_mean, runs max
```
With deterministic rollouts every pairwise total variation is 0 or 1, so **max() is binary by
construction** — it fires whenever ANY alternative differs. mean() instead gives the FRACTION of
alternatives that differ. Measured in-loop on budget routing:
```
aggregation   distinct values   at ceiling   gini
max                  2             76.5%     0.000     <- what every FL and CVRP run has used
mean                29             10.3%     0.262     <- graded and selective
```

**IMPLICATION FOR THE PAPER — needs a human decision.** FrozenLake is the paper's headline env
and its published numbers were produced by this max fallback. The results are not invalid (max is
a legitimate aggregation) but the paper DESCRIBES weighted_mean, which is not what ran. Either
relabel the method as max-aggregation, or re-run FL with true averaging and see if the win holds.
**Behaviour deliberately LEFT UNCHANGED** in code so `paper/repro/` still reproduces; the fallback
now raises a RuntimeWarning explaining itself. Do not "fix" it silently.

**SECOND KNOB, same root problem.** `cf_gamma=0.99` discounts the counterfactual return, so "same
customers served, different ORDER" scores differently — which destroys exactly the ties budget
mode exists to create. Corrected runs use `cf_gamma=1.0` so the return is the raw integer count.

**ACTION.** Launched sweep 273435 (720 runs: 4 budgets x 3 capacities {10,6,5} x 5 arms x 12
seeds) with `consequence_aggregation='mean'` + `cf_gamma=1.0`. Run prefix `bdm_`; the analysis
keeps it in a separate `meanagg` cell group so it is never pooled with the max runs. The four
earlier sweeps (273322/273382/273406/273411, 1,296 runs) all used the binary max score — they are
still a valid measurement OF MAX, and are the control this compares against.

**ALSO.** The main dial sweep finished 600/600 with zero failures. Analysis now refuses to compare
cells whose arms have unequal seed counts (an early partial read produced P(beats PER)=0.000 and
nan cells — the same "never read a sweep early" trap recorded for JaxNav).

### 2026-08-18 — BUDGET MODE built; 1,296-run dial sweep launched (results pending)
Attacking the CVRP Claim-2 null at its diagnosed cause rather than re-running it. Switched the
routing objective to the ORIENTEERING variant: serve as many customers as possible on a closed
tour within travel budget B. Reward becomes an INTEGER COUNT, so action outcomes can TIE and the
total-variation score stops saturating; B controls difficulty directly, so headroom is a knob.
Prior art is deep (distance-constrained VRP — Laporte/Desrochers/Nobert 1984; the Orienteering
Problem family), so this is a recognized OR variant, not an env invented to make CCE win.

**BUILT** (branch research/cce-cvrp-logistics, commits 9713876 / 7bf1f1d / 14cb90f — note the
pre-existing CVRP work was UNCOMMITTED and is now safe):
- `envs/routing_budget.py` — budget env. Distances quantized to integer units so budget-spent is
  an EXACT state variable (the instance is defined on those integers; the oracle is exact for it,
  not approximate). Action mask reserves the return leg, so the vehicle can never strand itself.
- `analysis/claim1/cvrp/budget_oracle.py` — exact max-servable DP. Spend strictly increases, so
  the state graph is a DAG and one backward pass solves it; Bellman-residual self-check included.
- `tests/test_routing_budget.py` — 18 tests. DP == BRUTE FORCE at six budget settings; the
  all-customers optimum matches permutation search. 39/39 including the old CVRP tests.
- `agents/cvrp/` build_env / evaluate / config / CLI wired. `opt_ratio` becomes served/max-servable
  so all downstream rliable machinery is unchanged.

**MEASURED BEFORE RUNNING (the pre-registration).** Exact-oracle stakes + a plain-DQN gate:
```
budget_mult   B(u)   states    optimal served   gini   dead%   DQN-uniform curve
   0.55        19     4,707        5/10         0.222  12.2    -
   0.75        26    47,582        8/10         0.214  11.4    0.750 -> 0.875, climbing
   0.95        33   183,826        9/10         0.262  16.3    0.889 flat from ep 400
   1.30        46   382,195       10/10         0.369  30.3    1.000 at ep 400  <- CEILING
FrozenLake 8x8 det (where CCE wins)             0.559  50.9
```
**MY FIRST PREDICTION WAS WRONG AND THE ORACLE CORRECTED IT.** I expected a TIGHT budget to
concentrate stakes. It does the opposite: when everything is on a knife edge ~88% of states have
stakes, which is the same flatness as before. Loose budgets concentrate stakes (gini 0.22 -> 0.37)
but destroy headroom. The two things CCE needs move in OPPOSITE directions on this dial, so the
registered prediction is an INVERTED U — advantage peaks mid-dial. Falsifiable three ways (flat,
monotone up, monotone down all contradict it).

**LAUNCHED — 1,296 runs, 4 SLURM arrays, all healthy at time of writing:**
- 273322 `budget_dial` 600 runs: 5 budgets x 2 capacities {10,6} x 5 arms x 12 seeds
- 273382 `budget_dial_cap5` 240 runs: capacity 5 (tightest feasible; max demand 4)
- 273406 `budget_dial_tuning` 96 runs: CCE knobs on the capacity-6 headroom cell —
  cf_n_rollouts=60 (the notebook's score-quantization caveat), mu=0.5, score_interval=5
- 273411 `budget_dial_ring12` 360 runs: SIZE axis, new 12-customer `ring12` instance
  (507k states, 1.8 GB peak, 9 of 12 servable at 0.80x)
Analysis: `analysis/claim2/cvrp_budget_sweep.py` (instance-aware; AUC, final, ep@thr, bootstrap
P(beats PER), dial figure). Plan: `plans/cce-cvrp-budget-mode.md`.

**EARLY PARTIAL READ (first cells only, NOT a result).** At budget 0.60x / capacity 10 every arm
reaches final 1.0000 and P(beats PER) sits at 0.39-0.50 — no CCE advantage at the tight end, which
is what the pre-registration expected. Capacity 6 does NOT ceiling (final 0.957), which is why the
cap-5 and ring12 sweeps exist.

**REJECTED / DEAD ENDS this session.** 13-customer instance: 4.4M states, 6.4 GB at 1.00x — too
heavy to build inside every run. `pkill -f <script>.py` killed its own shell (the pattern matched
the invoking command line) — use a narrower pattern or the job id.

**WATCH-ITEM.** In budget mode `opt_ratio` is COARSE (served is an integer, so the curve steps by
1/optimal). AUC over ~160 eval points is the mitigation. If arms still cannot be separated, that
is a measurement limit to report, NOT a null to claim.

### 2026-08-13 — NEW ENV: CVRP (logistics). **CLAIM 1 LANDED (3 seeds).** Claim 2 ruled out here.
Built a routing environment (`envs/cvrp.py`) as the 2nd good-oracle env C1 needed. Branch
`research/cce-cvrp-logistics` (worktree). Chose to REIMPLEMENT in JAX rather than install
Jumanji: jumanji+ortools are not installed, jax is 0.9.1, and installing risked breaking the
shared `counterfactual` env other live experiments use — same call the team made for FrozenLake
and DoorKey, and we need our own transition table for the oracle anyway.

**THE RESULT — Claim 1, 10 customers, capacity 10, 3 seeds, 1000 of 31345 decision states:**
```
              rho(CCE, exact oracle)      precision@10%
untrained       0.522 +/- 0.002              0.220
mid             0.668 +/- 0.012              0.263
trained         0.648 +/- 0.013              0.237     random chance = 0.100
```
All p < 1e-70. Seeds agree tightly. rho RISES untrained->mid (0.52->0.67), then DIPS slightly at
fully-trained in ALL 3 seeds (mid > trained every time). Differs from FrozenLake's monotone
0.319->0.765->0.889 — routing starts much higher (dense reward means even an untrained policy
senses the geometry) and peaks earlier. Figures: `docs/figures/real/claim1/cvrp/`.
CAVEAT: CCE scores are quantized in steps of 1/n_rollouts (=0.04 at 25 rollouts) — visible as
bands in the scatter; more rollouts would sharpen rho. ~14% of states still saturate at 1.0.

**ENV FACTS.** 10 stops + depot, demands sum 24 vs capacity 10 -> 3 loads, 13 decisions/episode,
37,918 states (5,122 without the load limit), obs = 22 features (one-hot node + served bits +
load fraction) NOT one-hot state (routing has too many states to memorize), 11 masked actions.
Oracle = exact backward induction, VALIDATED against brute force on both TSP (all permutations)
and CVRP (permutations x optimal load-split). Sanity gate: plain DQN-uniform reaches **1.0000 of
optimal** (random policy 0.62).

**CLAIM 2 — RAN IT. CLEAN NULL (50 runs: 5 arms x 5 seeds x traffic {0, 0.15}).**
```
TRAFFIC OFF      eps->0.95   AUC     final    P(beat PER, AUC)
uniform             1000    0.980   0.9949        0.758   <- random replay "beats" PER
PER                  750    0.978   0.9969         --
CCE-only            1000    0.978   0.9970        0.598
CCE+TD add           750    0.981   0.9979        0.760
CCE+TD mul           750    0.978   0.9954        0.516
TRAFFIC ON
uniform              750    0.973   0.9883        0.319
PER                 1000    0.975   0.9921         --
CCE-only            1500    0.971   0.9945        0.121   <- WORSE than PER
CCE+TD add           750    0.978   0.9927        0.680
CCE+TD mul          1250    0.970   0.9928        0.201   <- WORSE than PER
```
No arm reliably beats PER in either condition. Only CCE+TD(add) is ahead both times (AUC 0.981 /
0.978, P=0.76/0.68) — inside the noise at 5 seeds, gap 0.003. Uniform beating PER at traffic-off
is the tell that this is a noise-dominated measurement.

**THE REASON IS HEADROOM, NOT A DEAD SIGNAL — I had this wrong at first.**
- **CORRECTION.** An earlier entry in this session claimed "the TV score is a CONSTANT (1.0 at
  100% of states)". That came from an OFFLINE probe: uniformly-sampled states scored with a
  frozen, fully-trained greedy policy. **Instrumenting the actual training tells a different
  story**: scores computed in-loop have mean 0.841, std 0.125, 42 distinct values, only 35.9%
  saturated (traffic off); 0.781 / 0.133 / 45 / 20.1% (traffic on). CCE had a real, varying
  signal to prioritize with. The "flat constant" claim was a measurement artifact of the probe.
- **The sufficient reason for the null: the task is solved in ~750-1500 episodes** (we trained
  14,000). No replay strategy can matter when the problem is over that fast. Random replay tying
  PER is the proof. Same ceiling trap the graded-slip sweep hit at slip>=0.333.
- Traffic DID grade the score as predicted (saturation 36% -> 20%), it just didn't buy a win.

**Supporting (still true):** stakes are flat — gini 0.23 (cap 10) vs FrozenLake 0.56; max/median
2.7x vs 35.9x; 3% of states "dead" vs FrozenLake's 51%. Five geometries tried (ring / clustered /
hub+outliers / hub+1remote / two-lobes), gini stayed 0.19-0.27. Property of the domain, not the layout.

**IDEA NOT YET TRIED — BUDGET MODE (the way to get headroom).** Probe (400 states, offline):
re-scoring the same rollouts under a pass/fail fuel-budget outcome instead of continuous distance
moves gini 0.007 -> 0.43 and saturation 97.6% -> 15.4%. Build = one global budget B, distance-spent
added to the state (bins), reward 1 iff total <= B. Oracle stays exact ("can I still finish under
B from here?" is a reachability DP). Set B just above optimal so most policies FAIL -> headroom.
Untested; the offline probe is only a direction-check.

**SHARPENED THEORY RULE (the real contribution from this env).** Why did FL-det win but CVRP-det
die, when both are deterministic? Because TV between point masses is 0/1, so it is informative
only when returns TIE:
```
FrozenLake  reward 0/1, sparse -> most actions give the SAME return -> TV=0 -> fires on ~4%
                                  of states. SELECTIVE -> useful.
CVRP        reward = distance, continuous -> every action gives a DIFFERENT number -> TV=1
                                  -> fires on 100%. SATURATED -> useless.
```
So the CCE/TV score needs EITHER ties in the returns (sparse/discrete reward) OR stochasticity
(graded TV). Continuous + deterministic saturates it. This explains FL-det (win), FL-stoch (null)
and CVRP-det (dead) in one statement, and it is a testable extension of Corollary 1.

**`travel_noise` (traffic) added to the env** — multiplies each leg's cost by max(0,1+sigma*z).
Zero-mean, so env.P, the exact oracle and the optimal plan are UNCHANGED and a
deterministically-trained policy stays valid. Used ONLY for Claim 1 (needed: it turns the 0/1 TV
into a graded score — 17 distinct values at sigma=0.15 vs 1 at sigma=0). Standard OR variant
(stochastic travel times / VRPSTT), and the same move DoorKey made with `slip_prob`.
Claim 2 would use travel_noise=0.

**BUGS FOUND + FIXED**
- Oracle topological order: the CVRP reload edge (customer -> depot) keeps the visited-mask
  unchanged, so a popcount-only sort is NOT topological and silently produced wrong Q*. Fixed
  with a depot-first secondary key + a Bellman-residual self-check that raises on any bad order.
- `best.pkl` freezes once opt_ratio hits 1.0 (update is a strict `>`), so 'mid' and 'trained'
  resolved to the SAME weights and the first C1 run reported one measurement twice. `pick_checkpoints`
  now uses the LAST checkpoint for 'trained' and hashes the three to warn on duplicates.

**NOT DONE / NEXT.** No CCE-vs-PER training has ever been run here (only 200-episode crash tests
of all 5 arms — they run). Recommend: keep CVRP as a **Claim-1-only** environment; do not spend
cluster hours on a C2 sweep. Slide assets built: `cvrp_optimal_plan.gif`, `cvrp_learned_plan.gif`,
`cvrp_tsp_plan.gif`, `cvrp_how_it_works.gif`, `cvrp_teach_sar.gif`, `cvrp_env_spec_table.png`.
Plan: `plans/cce-cvrp-logistics.md`. All runs local CPU; nothing submitted to SLURM.

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
