# Lab Notebook — Counterfactual Reasoning (CCE)

Read first. LOG = append-only (never edit). Rewrite only STATUS/NEXT.
Before /clear: append a dated LOG entry, especially DEAD ENDS.

## GOAL
ICLR 2027 (iclr.cc). Deadline ~late Sep 2026 (est.; recheck). ~16 weeks from 2026-06-02.

## THE BAR
  C2 (speeds learning):   have FL-det ✓  → need 2 MORE clean scenarios (DoorKey-det is NULL, doesn't count)
  C1 (finds the moments): need deterministic-FL redo; the "+1 MORE good-oracle env" is now MET —
    DoorKey-slip, rho 0.43 p<0.0001 (3 seeds). See experiment/cce-doorkey branch.

## COVERAGE
```
             C1 (finds moments)          C2 (speeds learning)
FL-det       REDO (was slippery)          WIN ✓ mul 1.00 vs PER 0.46   (25 seeds)
FL-stoch     noise kills score            NULL 0.67–0.75               (10 seeds)
SMAX-3m      no oracle                    MARGINAL 0.722 vs 0.710      (10 seeds)
SMAX-8m      —                            UNFINISHED (never ran clean)
C4           —                            NOT FAIRLY TESTED (buggy)    <- DIG
Chess        oracle too weak              no improvement               DROPPED
DoorKey-slip WIN ✓ rho 0.43 p<.0001       n/a (needs slip; see -det)   (3 seeds)
DoorKey-det  n/a (degenerate w/o slip)    NULL all→100%, PER fastest   (10x5 seeds)
```

## STATUS (2026-06-02)
Data so far = verified + frozen in paper/repro/ (master 861c0e3).
NOT done experimenting — no claim hits its target scenario count yet.

## NEXT
1. DoorKey: add lava tiles (irreversible terminal, like FL holes) for real catastrophe
   structure — DoorKey has none, which is the diagnosed cause of the C2 null and the low
   C1 ceiling (rho 0.43 vs FL's 0.89). Re-test both claims after. Branch: experiment/cce-doorkey.
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
- Claim-1 `score_states.py` config-default trap: `agent.load(ckpt)` restores network weights
  ONLY, never the env/config the checkpoint was trained under. Constructing the agent bare
  (e.g. `FrozenLakeDQN()` / `DoorKeyDQN()`) silently falls back to DEFAULT_CONFIG for the CCE
  rollout env. Bit DoorKey for real (rollout env defaulted to slip_prob=0.0 while checkpoints
  were trained+oracled at 0.2, degenerating CCE scores into a coarse 0/1 signal — see
  2026-08-06 LOG). frozen_lake/score_states.py has the IDENTICAL bare-construction pattern and
  is only safe because DEFAULT_CONFIG's `is_slippery=True` happens to match what FL's C1 needs
  — worth a defensive explicit-config fix there too before that default ever changes.

## IDEAS NOT TRIED
- slip-robust metric (spread of MEAN returns, not TV) — may rescue stochastic null.
- adaptive μ (CCE early → TD late).
- C1 perturbation/causal test.
- add lava/catastrophe to DoorKey — see NEXT.

## SCOPE
Active: FL-det, FL-stoch, SMAX-3m, C4.   Dropped: Chess, raw diagnostics.

## LOG (append-only, newest on top)

### 2026-08-06 — DoorKey: 2nd good-oracle env for C1 (real win); real bug found+fixed; C2 null (real)
New branch `experiment/cce-doorkey`, split off `research/andon-vending-bench-cce` (that
worktree/branch was for the unrelated vending-bench idea — kept separate per branch-per-task).

Built `envs/doorkey.py`: tabular pure-JAX DoorKey-6x6, 154 enumerated states
(cell, dir, has_key, door_state), mirrors `frozen_lake.py` — exposes `env.P` for exact VI
since neither MiniGrid nor Navix expose a transition table. Deterministic by default; new
`slip_prob` knob makes it stochastic (mirrors FL's `is_slippery`). +1 terminal reward (not
MiniGrid's step-count-dependent reward, which would break the tabular oracle). Independently
audited by a from-scratch reference implementation: 0/1078 transition mismatches, oracle
matches an independent value iteration to 2e-13, 154/154 states reach the goal — env itself
is solid.

**Claim 1** (3 seeds, slip=0.2 — CCE's TV signal is degenerate without stochastic rollouts,
same reason FL's C1 uses the slippery map): rho 0.13 (untrained) → 0.45 (mid) → 0.43 (trained),
p<0.0001 from mid onward, all 3 seeds agree in direction. **Real, positive 2nd-domain result.**
Ceiling well below FL's 0.89 — diagnosed why: (a) DoorKey's oracle action-gaps have a narrow
dynamic range (no catastrophe → no hole-adjacent-style spike like FL) and (b) even a
100%-win-rate policy only agrees with Q* on 58–74% of states (it locks onto ONE good path and
never learns accurate Q off it), adding noise to CCE's rollout-based read elsewhere.

**Bug found + fixed** (real, not a theory issue): `analysis/claim1/doorkey/score_states.py`
built the CCE rollout env via `DoorKeyDQN()` with no config → silently defaulted to
`slip_prob=0.0` (DEFAULT_CONFIG, tuned for C2), NOT the training run's 0.2. `agent.load()`
restores weights only, never the env. Symptom: CCE scores degenerated to a coarse 0/1 signal
(spike-vs-spike TV) instead of a real distribution. Verified the fix in isolation (0/1 spikes
→ smooth 0.36–0.70 spread) before trusting the re-run; rho roughly doubled after the fix (was
0.15–0.20). See GOTCHAS — the identical bare-construction pattern exists in
`frozen_lake/score_states.py` and is only safe by accident of FL's default.

**Claim 2** (5 algorithms × 10 seeds, slip=0, deterministic): NULL, confirmed clean. All 50
seeds are genuine completions — 2 were killed early by the cluster (SIGKILL exit 0:9, no code
error, AvgR was still rising when cut off), resubmitted as jobs 271068/271069, both converged
100% at 11 steps. C2 was never affected by the C1 scoring bug (its trainers build their own
correctly-configured env from the sweep's own config). All 5 algorithms → 100% win rate; PER
fastest to 90% threshold (median 180k env steps); P(alg > PER) = 0.550 **uniformly** across all
3 CCE variants (driven by seed noise, not algorithm choice). Cause: DoorKey is fully reversible
and always solvable — no catastrophe → shallow Q-values → no large action-value gap for CCE to
exploit (Theorem 1's bound needs one). Same mechanism that makes FL's C1 need the slippery map
and FL's C2 win on the deterministic map — DoorKey just has no analog to FL's holes.

NEXT: add lava tiles to DoorKey (irreversible terminal, like FL holes) for real catastrophe
structure, re-test both claims.

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
