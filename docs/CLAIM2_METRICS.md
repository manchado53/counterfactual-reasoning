# Claim 2 — Metrics Cookbook

> **How we measure "CCE speeds up learning."** Definitions, formulas, and how-to-compute
> only. **No results or numbers here** — current results live in `/lab-notebook.md` (status)
> and `/paper/paper.tex` (the paper).

## What Claim 2 tests
CCE prioritization reaches competent play faster than the baselines, without hurting final
(asymptotic) performance.

## Algorithms compared (5)
| Label | Replay priority `p` for a transition |
|---|---|
| DQN-Uniform | uniform (no priority) |
| DQN+PER | `p = p_δ`  (\|TD error\|) — the baseline everything is compared to |
| DQN+CCE-only | `p = p_C`  (consequence only; = additive with μ=1) |
| CCE+TD additive | `p = μ·p_C + (1−μ)·p_δ`                       (Eq 4) |
| CCE+TD multiplicative | `p = p_C^{μ_C} · p_δ^{μ_δ} / Z`  (Eq 5; uniform fallback if all p→0) |

`p_C` = consequence priority (the Claim-1 score). `p_δ` = \|TD error\|.
Defaults: μ = 0.25, consequence_metric = total_variation.
Config→label mapping: `analysis/claim2/parse_logs.py`.

## Primary signal
**Win rate** ∈ [0,1] (fraction of eval episodes won / solved). Drives every metric below;
already normalized, so no rescaling needed.

## Metrics (rliable; 95% stratified bootstrap)
**1. IQM learning curves.** IQM win-rate across seeds at each checkpoint vs env steps, with
CI bands. IQM trims the top/bottom 25% of seeds (robust to outliers). One curve per algorithm
per environment — do **not** aggregate across environments.

**2. Final IQM.** IQM of each seed's mean over the **last 10% of checkpoints**. Tests whether
the early advantage costs asymptotic performance.

**3. P(improvement) over DQN+PER.** Probability a random seed of the variant beats PER on
final win rate. Directional only at small N (CIs ≈ ±10–15pp at N=10).

**4. Steps-to-threshold (optional — not in current paper).** First checkpoint with
win rate ≥ threshold; report median ± IQR across seeds; count seeds that never cross
(censored → ∞).
- **Threshold rule:** `70% × DQN+PER median final win rate`, **pre-registered** before
  seeing multi-seed results. 70% sits on the steep part of the curve (80–90% lands in the
  plateau, where every algorithm ties).

**5. Wall-clock (optional — appendix / honesty check).** From `timing.jsonl`:
`total` = the single `total` timer; CCE overhead = sum of `update.scoring.*` leaf timers;
training = total − scoring.

(Optional env-specific secondary signals, no rliable, separate plots: episode length; SMAX
allies-alive.)

## How to run
1. Submit the 5×N job set: `agents/<env>/run_experiments.py` → writes a manifest (job_id → config).
2. Analyze:
```
python -m counterfactual_rl.analysis.claim2.run_analysis --manifest <m> --env <e> --threshold <t>
```
`parse_logs` resolves each job's run dir (`agents/<env>/runs`, then legacy `agents/shared/runs`),
builds `(n_seeds, 1, n_checkpoints)` arrays, computes the metrics, and writes fig1/fig2/fig4b
(plus optional fig3 steps-to-threshold / fig5 wall-clock).

## Pre-flight (per env)
Before trusting any Claim-2 run, clear the **PRE-FLIGHT CHECKLIST** in `/lab-notebook.md`
(rewards[agent_player], crossing-logic triggers, LeakyReLU, full-length horizon,
total_variation, and the plain-DQN sanity check first).
