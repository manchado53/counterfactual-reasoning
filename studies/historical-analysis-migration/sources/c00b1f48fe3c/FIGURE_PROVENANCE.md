# Figure Provenance — `paper/figures/`

Purpose: for every PNG in this folder, record **what script made it, from which
runs, with which seeds and config**, so any figure can be traced and regenerated.

Last traced: 2026-06-02. Source of truth for the manuscript: `paper/paper.tex`.

> **How the paper figures get here:** every figure is generated into
> `docs/figures/real/...` by an analysis script, then **copied** into
> `paper/figures/`. To regenerate: re-run the analysis command (writes to
> `docs/figures/real/...`), then copy the PNG over.

---

## 1. Quick map (every file in this folder)

| File | Claim / Env | Generator | Data source (manifest / checkpoints) | Cited in paper.tex? |
|---|---|---|---|---|
| `fig_c1_scatter_stages.png` | C1 / FrozenLake (slippery) | `analysis/claim1/frozen_lake/run_analysis.py` | C1 checkpoints seed 0/1/2 | **yes** (fig:c1_scatter) |
| `fig_c2_grid_heatmaps.png` | C1 / FrozenLake (slippery) | `analysis/claim1/frozen_lake/run_analysis.py` | C1 checkpoints seed 0 | **yes** (fig:c2_heatmaps) |
| `fig_c4_precision_at_k.png` | C1 / FrozenLake (slippery) | `analysis/claim1/frozen_lake/run_analysis.py` | C1 checkpoints seed 0/1/2 | **yes** (fig:c4_precision) |
| `fig_c1_rho_table.png` | C1 / FrozenLake (slippery) | `analysis/claim1/frozen_lake/make_table.py` | real ρ values, stored as a literal in the script (see §2) | no (orphan) |
| `fig1_iqm_frozen_lake_no_slip.png` | C2 / FL deterministic | `analysis/claim2/run_analysis.py` | `claim2_no_slip_2026-05-09.json` | **yes** (fig:fl_det_iqm) |
| `fig2_final_iqm_frozen_lake_no_slip.png` | C2 / FL deterministic | `analysis/claim2/run_analysis.py` | `claim2_no_slip_2026-05-09.json` | **yes** (fig:fl_det_final_iqm) |
| `fig4b_prob_improve_curves_frozen_lake_no_slip.png` | C2 / FL deterministic | `analysis/claim2/run_analysis.py` | `claim2_no_slip_2026-05-09.json` | **yes** (fig:fl_det_prob) |
| `fig3_steps_thresh_frozen_lake_no_slip.png` | C2 / FL deterministic | `analysis/claim2/run_analysis.py` | `claim2_no_slip_2026-05-09.json` | no (orphan) |
| `fig5a_wallclock_frozen_lake_no_slip.png` | C2 / FL deterministic | `analysis/claim2/run_analysis.py` | `claim2_no_slip_2026-05-09.json` + `timing.jsonl` | no (orphan) |
| `fig1_iqm_frozen_lake.png` | C2 / FL stochastic | `analysis/claim2/run_analysis.py` | `claim2_main_merged.json` | **yes** (fig:fl_stoch_iqm) |
| `fig2_final_iqm_frozen_lake.png` | C2 / FL stochastic | `analysis/claim2/run_analysis.py` | `claim2_main_merged.json` | **yes** (fig:fl_stoch_final_iqm) |
| `fig4b_prob_improve_curves_frozen_lake.png` | C2 / FL stochastic | `analysis/claim2/run_analysis.py` | `claim2_main_merged.json` | **yes** (fig:fl_stoch_prob) |
| `fig1_iqm_smax_3m.png` | C2 / SMAX 3m | `analysis/claim2/run_analysis.py` | `claim2_main_3m_2026-05-07.json` | **yes** (fig:smax_iqm) |
| `fig2_final_iqm_smax_3m.png` | C2 / SMAX 3m | `analysis/claim2/run_analysis.py` | `claim2_main_3m_2026-05-07.json` | **yes** (fig:smax_final_iqm) |
| `fig4b_prob_improve_curves_smax_3m.png` | C2 / SMAX 3m | `analysis/claim2/run_analysis.py` | `claim2_main_3m_2026-05-07.json` | **yes** (fig:smax_prob) |
| `fig3_steps_thresh_smax_3m.png` | C2 / SMAX 3m | `analysis/claim2/run_analysis.py` | `claim2_main_3m_2026-05-07.json` | no (orphan) |
| `fig5a_wallclock_smax_3m.png` | C2 / SMAX 3m | `analysis/claim2/run_analysis.py` | `claim2_main_3m_2026-05-07.json` + `timing.jsonl` | no (orphan) |

**5 orphan figures** are present in this folder but NOT referenced by `paper.tex`:
`fig_c1_rho_table.png`, `fig3_steps_thresh_*` (×2), `fig5a_wallclock_*` (×2).

All manifest paths are under
`src/counterfactual_rl/agents/<env>/experiments/<month>/<name>/<name>.json`.
A manifest maps `SLURM_job_id → per-run config override dict`. Each run's raw
metrics live in `src/counterfactual_rl/agents/shared/runs/<job_id>/metrics.log`
(+ `timing.jsonl`).

---

## 2. Claim 1 — FrozenLake (figures `fig_c1_*`, `fig_c2_grid_heatmaps`, `fig_c4_*`)

**Environment: SLIPPERY (stochastic) FrozenLake 8×8.** Both the oracle and the
CCE rollouts are slippery (see caveat below).

**Generator (scatter / heatmaps / precision):**
```
PYTHONPATH=src python -m counterfactual_rl.analysis.claim1.frozen_lake.run_analysis
```
Defaults (all baked into `run_analysis.py:116-120`):
`--seeds 0 1 2  --metric total_variation  --n-rollouts 100  --horizon 500  --gamma 1.0`

- **Oracle** (`oracle.py:14`): exact value iteration on the known transition
  table, `compute_oracle(map_name='8x8', is_slippery=True, gamma=0.99)`.
  Label per state = mean |Q*(s,a*) − Q*(s,a)| over a≠a*. 53 non-terminal states.
- **CCE score** (`score_states.py`): total-variation among per-action return
  distributions from rollouts under the checkpoint's own greedy policy. The
  rollout env comes from the checkpoint → slippery.
- **Output:** `docs/figures/real/claim1/frozen_lake/` → copied to `paper/figures/`.

**Checkpoints scored:** `src/counterfactual_rl/analysis/claim1/frozen_lake/checkpoints/seed_{0,1,2}/{untrained,mid,trained}.pkl`
(stages = untrained ≈ ep 150, mid ≈ ep 3900, trained = best).

**Where those checkpoints came from (training):**
manifest `frozen_lake/.../claim1_dqn_2026-05-06/claim1_dqn_2026-05-06.json` —
3 jobs (255145, 255146, 255147), config:
`algorithm=dqn` (PER), `map_name=8x8`, `n_episodes=15000`,
`early_stop_win_rate=0.99`, `seed ∈ {0,1,2}`, `is_slippery` not overridden →
**defaults to True** (`agents/frozen_lake/config.py`).
*(Link between these jobs and the staged .pkl files is by date/seed match;
the stage-extraction step is not scripted in-repo — confirm if needed.)*

**rho table note (`fig_c1_rho_table.png`):** the numbers are **real results** from
a `run_analysis` run (trained ρ = [0.849, 0.926, 0.891] → mean 0.889). They are
stored as a literal `DATA` dict in `make_table.py` (lines 13-17), so the plot
renders from those stored values rather than recomputing the rollouts each time.
If C1 is re-scored (e.g. on deterministic FL), update this dict to match.

> **CAVEAT — env mismatch.** Claim 1 here is on **slippery** FL, but the Claim-2
> headline win is on **deterministic** FL. The two are different environments.
> Flagged for the paper revision.

---

## 3. Claim 2 — common pipeline

**Generator (all C2 figures):**
```
ANALYSIS_MANIFEST=<manifest.json> ANALYSIS_ENV=<env> ANALYSIS_THRESHOLD=<t> \
ANALYSIS_OUT=docs/figures/real/claim2/<env> \
sbatch src/counterfactual_rl/analysis/claim2/run_analysis.sh
# equivalently, direct:
PYTHONPATH=src python -m counterfactual_rl.analysis.claim2.run_analysis \
    --manifest <manifest.json> --env <env> --threshold <t> \
    --out docs/figures/real/claim2/<env> --reps 50000
```
Figure filenames are suffixed with `--env`. Only `fig1_iqm`, `fig2_final_iqm`,
`fig4b_prob_improve_curves` are used by the paper; `fig3_steps_thresh`,
`fig5a_wallclock` (and `fig4`, `fig5b`, `fig5c`, `fig_length`, `fig_allies`) are
extra. Threshold affects ONLY `fig3`/`fig5b`, so it does not change the paper's
chosen figures.

**Metrics:** rliable, 95% stratified bootstrap, default 50000 resamples
(`compute_metrics.py`). IQM curves, final IQM (last 10% of checkpoints),
P(alg > DQN+PER).

**Algorithm label mapping** (`parse_logs.py:191-220`):

| manifest config | paper label |
|---|---|
| `algorithm=dqn-uniform` | DQN-Uniform |
| `algorithm=dqn` | DQN+PER |
| `consequence-dqn`, `additive`, `mu≥1.0` | DQN+CCE-only |
| `consequence-dqn`, `additive`, `mu<1.0` | CCE+TD (add) |
| `consequence-dqn`, `multiplicative` | CCE+TD (mul) |

---

## 4. Claim 2 — FrozenLake DETERMINISTIC (`*_frozen_lake_no_slip`)

- **Manifest:** `frozen_lake/experiments/2026-05/claim2_no_slip_2026-05-09/claim2_no_slip_2026-05-09.json`
- **Runs:** 125 jobs = **25 seeds (0–24) × 5 algorithms.**
- **Per-run config:**
  `map_name=8x8`, `is_slippery=False`, `n_episodes=15000`,
  `consequence_metric=total_variation`, `mu=0.25` (add/mul) or `1.0` (cce-only),
  `epsilon_decay_episodes=7500`, `score_interval=100`, `vectorized=True`,
  `cf_horizon=200`, `early_stop_win_rate=0.95`.
- **Analysis:** `--env frozen_lake_no_slip --threshold 0.75`
  (0.75 is the registered default in `run_analysis.py:46`).
- **Output:** `docs/figures/real/claim2/FL_deterministic/` → `paper/figures/`.
- **Note:** no SLURM analysis log for this env was found in `claim2/logs/`; the
  May-13 figure set was likely regenerated interactively. Reproduce with the
  command above.

---

## 5. Claim 2 — FrozenLake STOCHASTIC / slippery (`*_frozen_lake`)

- **Manifest:** `frozen_lake/experiments/2026-05/claim2_main_merged.json`
  (a merge of `claim2_main_2026-05-06` + `claim2_cce_multiplicative_2026-05-07`).
- **Runs:** 50 jobs = **10 seeds (0–9) × 5 algorithms.**
- **Per-run config:**
  `map_name=8x8`, **`is_slippery` not set → defaults to True (slippery)**,
  `n_episodes=15000`, `mu=0.25`, `consequence_metric=total_variation`,
  `epsilon_decay_episodes=7500`, `score_interval=100`, `vectorized=True`.
- **Analysis:** `--env frozen_lake --threshold 0.75`
  (logs `analysis_255559` failed; `analysis_255560` succeeded).
  Output: `docs/figures/real/claim2/frozen_lake/` → `paper/figures/`.

---

## 6. Claim 2 — SMAX 3m (`*_smax_3m`)

- **Manifest:** `smax/experiments/2026-05/claim2_main_3m_2026-05-07/claim2_main_3m_2026-05-07.json`
- **Runs:** 50 jobs = **10 seeds (0–9) × 5 algorithms.**
- **Per-run config:**
  `scenario=3m`, `n_episodes=25000`, `mu=0.25`,
  `consequence_metric=total_variation`, `epsilon_decay_episodes=10000`,
  `score_interval=200`.
- **Analysis:** `--env smax_3m --threshold 0.60` (registered default).
  Output: `docs/figures/real/claim2/smax_3m/` (also `docs/figures/real/smax_3m/`).
- **Note:** the SLURM analysis job `257322` (2026-05-09) was **CANCELLED on a time
  limit**; the published May-13 SMAX-3m figures were therefore regenerated by a
  later/local run not captured in `claim2/logs/`. Reproduce with the command above.

---

## 7. Verification status (checked 2026-06-02)

Run dirs resolve via `parse_logs._find_run_dir`: **per-agent `agents/<env>/runs/`
first, then legacy `agents/shared/runs/`.**

| Dataset | Manifest jobs | metrics.log present | Location | Notes |
|---|---|---|---|---|
| FL deterministic | 125 | **124/125** | `frozen_lake/runs` | missing job `257574` (CCE+TD-add). Checkpoint counts uneven (6–32) because `early_stop_win_rate=0.95` stops solved runs early; `parse_logs` pads. |
| FL stochastic | 50 | **50/50** | `frozen_lake/runs` | 34–38 checkpoints each, consistent. |
| SMAX 3m | 50 | **50/50** | `shared/runs` (legacy) | 250 checkpoints each, perfectly consistent. |

All three datasets are present and reproducible from their manifests.

**Remaining open items:**
1. FL-deterministic CCE+TD-add reproduces with 24/25 seeds (job `257574` gone). Re-run
   that one seed if a full 25 is wanted, else note n=24 for that algorithm.
2. The `fig_c1_rho_table` values are real but stored as a literal in `make_table.py`
   (see §2) — update the dict if C1 is ever re-scored.
3. Decide whether Claim 1 should be recomputed on **deterministic** FL so it
   matches the Claim-2 environment (currently slippery — §2 caveat).
4. The mock generators `docs/mock_claim1_figures.py` / `docs/mock_claim2_figures.py`
   write fabricated stand-ins to `docs/figures/claim1/` and `docs/figures/claim2_*`.
   These are NOT the paper figures — do not confuse them with the real outputs.
