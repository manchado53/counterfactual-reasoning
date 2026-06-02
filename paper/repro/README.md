# `paper/repro/` — self-contained reproduction bundle

Everything needed to rebuild the paper's figures **without the ~260 GB of raw
run directories**. All files here are tiny (total ~2 MB) and git-tracked, so the
paper survives even if the runs get purged from the (currently 100%-full) disk.

Full per-figure provenance (which script, which seeds, which config) lives in
`../figures/FIGURE_PROVENANCE.md`. This README is the *how to regenerate* side.

## Layout

```
paper/repro/
  manifests/        frozen snapshots of the job_id -> config maps
    claim2_no_slip_2026-05-09.json     FL deterministic  (125 jobs, 25 seeds x 5 alg)
    claim2_main_merged.json            FL stochastic     ( 50 jobs, 10 seeds x 5 alg)
    claim2_main_3m_2026-05-07.json     SMAX 3m           ( 50 jobs, 10 seeds x 5 alg)
    claim1_dqn_2026-05-06.json         C1 checkpoint training source (3 seeds)
  cache/
    claim2_frozen_lake_no_slip.npz     parsed rliable arrays (FL det)
    claim2_frozen_lake.npz             parsed rliable arrays (FL stoch)
    claim2_smax_3m.npz                 parsed rliable arrays (SMAX 3m)
    claim1_frozen_lake_oracle.npz      exact Q* oracle (value iteration)
    checkpoints/seed_{0,1,2}/{untrained,mid,trained}.pkl   C1 scored checkpoints
  build_cache.py     rebuilds cache/*.npz FROM the raw runs (only needed if re-deriving)
  replot.py          rebuilds the Claim-2 figures FROM cache/ alone (no runs needed)
  regen/             scratch output of replot.py (compare against ../figures/)
```

## Regenerate the Claim-2 figures (no raw runs needed)

```
PYTHONPATH=src python paper/repro/replot.py --reps 50000
```
- Reads only `cache/*.npz`. Writes `fig1_iqm_*`, `fig2_final_iqm_*`,
  `fig4b_prob_improve_curves_*` for all three envs into `regen/`.
- `--reps 50000` reproduces the published figures exactly; a smaller value is
  fine for a quick check (point estimates are reps-independent; only the
  bootstrap CIs widen).
- These three figure types per env are the ones used by `paper.tex`.

Sanity values it should print (point estimates, reps-independent):
- FL deterministic final IQM: CCE+TD(mul) ≈ 1.00, CCE+TD(add) ≈ 0.83,
  DQN+CCE-only ≈ 0.62, DQN+PER ≈ 0.46, DQN-Uniform ≈ 0.31.
- FL stochastic final IQM: all clustered ≈ 0.67–0.75 (the deliberate null).

## Regenerate the Claim-1 figures

Claim 1 needs the policy checkpoints (rollouts are run under them):
```
# checkpoints are snapshotted in cache/checkpoints/; the analysis script reads
# them from src/.../analysis/claim1/frozen_lake/checkpoints/, so restore if gone:
cp -r paper/repro/cache/checkpoints/seed_* \
      src/counterfactual_rl/analysis/claim1/frozen_lake/checkpoints/

PYTHONPATH=src python -m counterfactual_rl.analysis.claim1.frozen_lake.run_analysis
```
- Produces `fig_c1_scatter_stages`, `fig_c2_grid_heatmaps`, `fig_c4_precision_at_k`
  into `docs/figures/real/claim1/frozen_lake/`.
- The exact oracle is also cached in `cache/claim1_frozen_lake_oracle.npz`
  (map 8x8, **slippery**, gamma 0.99) if you want to reuse it directly.
- The rho table (`fig_c1_rho_table.png`) is drawn from real values stored as a
  literal in `make_table.py`; regenerate with
  `python -m counterfactual_rl.analysis.claim1.frozen_lake.make_table`.

## Rebuild the cache itself (only if re-deriving from raw runs)

```
PYTHONPATH=src python paper/repro/build_cache.py
```
Requires the raw run dirs (resolved via `parse_logs._find_run_dir`:
`agents/<env>/runs/` first, then legacy `agents/shared/runs/`). Not needed for
normal figure regeneration — the committed `cache/*.npz` already hold the numbers.

## Algorithm label mapping (manifest config -> paper label)

| manifest config | label |
|---|---|
| `dqn-uniform` | DQN-Uniform |
| `dqn` | DQN+PER |
| `consequence-dqn`, additive, `mu>=1.0` | DQN+CCE-only |
| `consequence-dqn`, additive, `mu<1.0` | CCE+TD (add) |
| `consequence-dqn`, multiplicative | CCE+TD (mul) |

## Notes
- Snapshots, not moves: nothing was removed from its original location.
- FL deterministic CCE+TD(add) has 24/25 seeds (one run, job 257574, was gone);
  recorded faithfully in the cache.
- `regen/` is throwaway verification output; the figures the paper compiles are
  the committed ones in `../figures/`.
