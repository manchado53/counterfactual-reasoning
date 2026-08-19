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
CVRP        WIN ✓ ρ .52→.67→.65        NULL — measured, 50 runs
(logistics)  exact oracle, 3 seeds      (solved in 750 ep → no headroom)
```

## STATUS (2026-06-02)
Data so far = verified + frozen in paper/repro/ (master 861c0e3).
NOT done experimenting — no claim hits its target scenario count yet.

## NEXT
1. C4 Layer 1 — verify fixes are in code, then: does plain DQN beat the opponent at all?
2. Redo C1 on deterministic FL.
3. Fix 3 paper.tex numbers (FL-det → 25 seeds; FL-stoch → 0.67–0.75; SMAX PER → 0.71).
4. Recheck ICLR 2027 deadline.

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
