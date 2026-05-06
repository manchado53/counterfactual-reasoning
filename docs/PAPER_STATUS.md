# Paper Status

**Last updated:** 2026-05-07

---

## Experiment Status

### Claim 2 — Sample Efficiency

| Environment | Metric | μ | Seeds | Status | Notes |
|---|---|---|---|---|---|
| SMAX 3m | total_variation | 0.25 | 10 | 🟡 Ready to submit | Config fixed; 50 jobs |
| SMAX 8m | total_variation | 0.25 | 10 | 🟡 Ready to submit | Config fixed; 50 jobs |
| FrozenLake 8×8 | total_variation | 0.25 | 10 | 🟡 Ready to submit | No pilot needed; threshold set post-training; 50 jobs |
| Gardner Chess | total_variation | 0.25 | TBD | ⏳ Not submitted | Seeds TBD (10 or 5); threshold set post-training |

**Submit commands:**
```bash
# From src/counterfactual_rl/agents/smax/
python run_experiments.py claim2_main_3m --max-concurrent 4   # 50 jobs, ~1.5h each on T4
python run_experiments.py claim2_main_8m --max-concurrent 4   # 50 jobs, ~2h each on T4

# From src/counterfactual_rl/agents/frozen_lake/
python run_experiments.py claim2_main --max-concurrent 4      # 50 jobs

# From src/counterfactual_rl/agents/chess/  (after seeds decision)
python run_experiments.py claim2_main --max-concurrent 4
```

### Claim 1 — CCE Identifies Consequential Moments

| Validator | Requires | Status | Notes |
|---|---|---|---|
| FrozenLake ΔQ oracle | 1 CCE FL seed with scored buffer | ⏳ Blocked on FL training | Value iteration code not yet written |
| Chess oracle (Δv) | 1 CCE chess seed with scored buffer | ⏳ Blocked on chess training | pgx value head available via `pgx.make_baseline_model` |

### Hyperparameter Sweep (completed)

| Phase | Winner | Seeds | Status |
|---|---|---|---|
| Phase 1 — metric sweep | total_variation (69.2% win rate) | 3 per metric | ✅ Done |
| Phase 2 — μ sweep | μ=0.25 (70.8% win rate) | 3 per value | ✅ Done |

Sweep figures: `src/counterfactual_rl/agents/smax/experiments/2026-03/metric_sweep_2026-03-01/` and `mu_sweep_2026-03-02/`

---

## Figures Status

### Claim 2 figures (multi-environment, generated once all envs complete)

| Figure | Description | Status |
|---|---|---|
| Fig 1 | 2×2 IQM learning curve grid | ⏳ Pending all environments |
| Fig 2 | Final IQM + P(improvement) bar chart | ⏳ Pending all environments |
| Fig 3 | Wall-clock breakdown across environments | ⏳ Pending all environments |
| Table 2 | Steps-to-threshold (3m filled, others pending) | 🔶 Partial — 3m only |

### Claim 1 figures

| Figure | Description | Status |
|---|---|---|
| FL scatter | CCE score vs ΔQ, Spearman ρ | ⏳ Pending FL training |
| Chess scatter | CCE score vs Δv, Spearman ρ | ⏳ Pending chess training |

---

## Paper Sections Status

| Section | Status | Notes |
|---|---|---|
| Abstract | ❌ TODO | Write last |
| 1. Introduction | ✅ Done | |
| 2. Related Work | ✅ Done | |
| 3. Background | ✅ Done | |
| 4. Consequence Prioritization | ✅ Done | |
| 5.1 Environments | ✅ Done | Facts verified against docs |
| 5.2 Algorithms | ✅ Done | |
| 5.3 Claim 1 | ✅ Done | SMAX hindsight dropped; FL + chess oracle only |
| 5.4 Claim 2 | ✅ Done | Sweep correctly documented (TV + μ=0.25) |
| 5.5 Implementation Details | ✅ Done | |
| 6. Results — Learning Curves | 🔶 Placeholder | Fig 1 placeholder in place |
| 6. Results — Steps-to-Threshold | 🔶 Partial | 3m data filled; others pending |
| 6. Results — Aggregate Metrics | 🔶 Placeholder | Fig 2 placeholder in place |
| 6. Results — Wall-Clock | 🔶 Placeholder | Fig 3 placeholder in place |
| 6. Results — Discussion | 🔶 Partial | 3m written; 8m/chess/FL stubs |
| 7. Future Work | ❌ TODO | |
| 8. Conclusion | ❌ TODO | |
| Appendix — Sweep tables | ⏳ Optional | Include if venue requires it |

---

## TODO List

### Urgent (blocking experiments)
- [ ] **Rerun SMAX 3m** with `total_variation` + `μ=0.25` (50 jobs) — ready to submit
- [ ] **Submit SMAX 8m** with `total_variation` + `μ=0.25` (50 jobs) — ready to submit
- [ ] **Submit FrozenLake claim2_main** (10 seeds × 5 algos = 50 jobs) — ready to submit; threshold set post-training
- [ ] **Decide Chess seeds** (10 or 5) then submit Chess claim2_main
- [ ] **Set FL threshold** after training — look at learning curves, pick win rate at ~70% of training
- [ ] **Set Chess threshold** after training — same process

### Analysis (run after experiments finish)
- [ ] Run `run_analysis.py` on 3m TV results → regenerate all 3m figures
- [ ] Run `run_analysis.py` on 8m results
- [ ] Run `run_analysis.py` on FL results
- [ ] Run `run_analysis.py` on Chess results
- [ ] Write FrozenLake ΔQ oracle analysis script (value iteration + Spearman ρ)
- [ ] Write Chess oracle analysis script (pgx value head + Spearman ρ)

### Paper writing (unblock as results arrive)
- [ ] Fill Steps-to-Threshold table (Table 2) with 8m, Chess, FL results
- [ ] Replace Fig 1/2/3 placeholders with real multi-environment figures
- [ ] Write 8m, Chess, FL discussion paragraphs in Results §6.5
- [ ] Write Claim 1 results paragraph + add scatter plot figures
- [ ] Write Future Work section
- [ ] Write Conclusion section
- [ ] Write Abstract (last)

### Code
- [ ] Write multi-environment plot script (`plot_figures_all.py`) for Fig 1/2/3
- [ ] Write FrozenLake value iteration + oracle correlation script
- [ ] Write Chess oracle correlation script

---

## Key Decisions Log

| Date | Decision |
|---|---|
| 2026-05-06 | Dropped SMAX hindsight validator from Claim 1 — keeping FL ΔQ + chess oracle only |
| 2026-05-06 | Corrected metric from wasserstein → total_variation per sweep results |
| 2026-05-06 | Restructured Results to 3 figures + 2 tables (not per-environment figure flood) |
| 2026-05-06 | Wall-clock fix: `parse_wallclock()` now uses `total` field as ground truth; scoring overhead from `update.scoring.*` leaf timers only |
| 2026-05-07 | No pilot runs needed — threshold is set post-training by inspecting learning curves |
| 2026-05-07 | FrozenLake stays at 10 seeds (same as SMAX) — sufficient signal, environment is stochastic |
| 2026-05-07 | Implemented SLURM throttle (`--max-concurrent N`) as shared utility; `--nice=10000` on all training scripts |
| 2026-05-07 | SMAX 8m prioritized below FL + Chess — env diversity stronger argument than same-env scaling |
