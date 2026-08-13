# Repo Cleanup Inventory (2026-06-01)

Nothing here has been deleted. Each group has an ID — tell Claude which IDs to remove.

## A. Run outputs / checkpoints (DISK ONLY — none are in git)

Total ~260 GB. All under gitignored `runs/`, so deleting is pure disk cleanup (no history change).

| ID | Path | Size | Notes |
|----|------|------|-------|
| R1 | `src/counterfactual_rl/agents/shared/runs` | 249 GB | 1,136 folders. 839 older than 2026-05-17, 297 newer. Newest 2026-05-30. |
| R2 | `src/counterfactual_rl/agents/frozen_lake/runs` | 5.7 GB | |
| R3 | `src/counterfactual_rl/agents/smax/runs` | 4.7 GB | |
| R4 | `src/counterfactual_rl/simulations/runs` | 361 MB | |
| R5 | `runs/` (repo root) | 253 MB | |
| R6 | `src/counterfactual_rl/agents/chess/runs` | 48 MB | |

One-liners (run yourself; review first):
```bash
# See sizes
du -sh src/counterfactual_rl/agents/*/runs runs src/counterfactual_rl/simulations/runs

# Nuke ALL run folders (~260 GB)
find . -type d -name runs -prune -exec rm -rf {} +

# Keep only runs touched in last 14 days in the big dir
find src/counterfactual_rl/agents/shared/runs -maxdepth 1 -type d ! -newermt 2026-05-17 -exec rm -rf {} +
```

## B. Tracked clutter (IN git — removing = a commit)

| ID | Files | Size | Verdict |
|----|-------|------|---------|
| T1 | `paper.aux`, `paper.fls`, `paper.fdb_latexmk`, `paper.log` | ~24 KB | LaTeX build junk. Delete + gitignore. |
| T2 | root `paper.md`, `references.bib`, `IEEEtran.cls` | ~315 KB | Stale dupes — real paper is in `paper/` (newer, has `.tex`). Root `paper.tex` already deleted in working tree. |
| T3 | `Counterfactual_RL (3).pdf` | 264 KB | Old PDF export (Feb 16). `paper/` has (10),(11). |
| T4 | `bugs.md`, `SMAC_OPTIMIZATION_FIXES.md`, `Sampling_Experiment (2).ipynb` | ~28 KB | Old notes / scratch notebook. |
| T5 | `bootstrap_visualization.{py,png}`, `gardner_chess.{gif,svg}` | ~1 MB | Loose figures at root; belong in docs/figures or trash. |

## C. Untracked scratch (NOT in git — just `rm`)

| ID | Files | Notes |
|----|-------|-------|
| U1 | `test_c4_diag.py`, `test_c4_obs.py` | Connect-four debug scripts at root. |
| U2 | `benchmark_mcts_opponent.{py,sh}` | Recent (May 29) — maybe still useful; confirm before deleting. |
| U3 | 22 `__pycache__` dirs + 128 `.pyc` | Build cache. Safe. `find . -name __pycache__ -prune -exec rm -rf {} +` |

## D. Untracked REAL work — should be COMMITTED, not deleted

`connect_four/`, `analysis/claim1/`, `analysis/diagnostics/`, `paper/`, and many
`docs/` additions are new work that isn't in git yet. Recommend committing these.

## E. Legacy / dead-code candidates (need care — some still imported)

| ID | Target | Imported by | Risk |
|----|--------|-------------|------|
| L1 | `environments/` (Gymnasium SMAC pipeline) | `analysis/counterfactual.py` (core!), `analysis/multidiscrete_counterfactual.py`, `simulations/smac_*` | MEDIUM — not fully dead. |
| L2 | `visualization/smac_plots.py`, `utils/smac_data_structures.py`, `utils/action_names.py`, `simulations/smac_*.py`, `examples/smac_*` | SMAC-only chain | LOW–MED — dormant; verify no live path. |

## F. Code refactor opportunities (no deletion; future work)

- 4× near-duplicate DQN impls (chess/fl/smax/connect_four) → shared base.
- 4× `consequence_dqn.py` variants → shared base (~80% identical).
- `smax_counterfactual.py` vs `smax_vectorized_counterfactual.py` overlap.
