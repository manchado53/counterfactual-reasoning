# Claim 1 — what each directory holds

`fig_claim1_summary.png` is the cross-environment figure. Rebuild:

```
python -m counterfactual_rl.analysis.claim1.claim1_summary --out docs/figures/real/claim1
```

It reads the routing numbers from `cvrp_final_{mean,weighted_mean}/` at import time, so the
figure cannot drift from the data. The FrozenLake numbers are hardcoded, with the command that
produces them in the file header — the FL audit reads `paper/repro/cache/checkpoints` and
writes no JSON.

## Routing (CVRP) result sets

| directory | seeds | aggregation | what it is |
|---|---|---|---|
| `cvrp_final_mean/` | 10 | `mean` | **HEADLINE.** ρ 0.576 → 0.765 → 0.741, precision@10% 0.367 / 0.478 / 0.454 |
| `cvrp_final_weighted_mean/` | 10 | `max` (see below) | the same 10 seeds under the aggregation that actually shipped — ρ 0.536 → 0.693 → 0.667 |
| `cvrp/` | 3 | `max` | the **first** routing Claim-1 result (2026-08-13, horizon 35). Superseded by `cvrp_final_mean/`. Kept because the paper figure, env spec table and the GIFs live here. |

`cvrp/fig_c1_paper_cvrp.png` is the 4-panel paper figure and is built from the **10-seed**
directories, not from the 3-seed JSON sitting beside it. Rebuild:

```
python -m counterfactual_rl.analysis.claim1.cvrp.paper_figure \
    --mean-dir docs/figures/real/claim1/cvrp_final_mean \
    --max-dir  docs/figures/real/claim1/cvrp_final_weighted_mean \
    --out      docs/figures/real/claim1/cvrp
```

Verified to reproduce byte-identically from the committed `pairs_*.npz` and JSON.

## Why `weighted_mean` means `max`

`analysis/metrics.py::compute_consequence_metric` honours `'weighted_mean'` only when
`action_probs` is supplied. FrozenLake and routing never supply it, so control falls through
to `max()`. The behaviour is deliberately UNCHANGED so `paper/repro/` keeps reproducing; the
fallback now raises a `RuntimeWarning`, and the Claim-1 scorers take an explicit `aggregation`
argument so an analysis can ask for `mean` without changing what training does.

## FrozenLake

`frozen_lake/*.png` are the paper's published figures and are **not** regenerated here — the
2026-08-23 audit confirmed they reproduce from `paper/repro/cache/checkpoints` to three
decimals. Re-running the FL analysis overwrites them, so redirect the output if you do.
