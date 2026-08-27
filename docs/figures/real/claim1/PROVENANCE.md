# Provenance — Claim 1

Same purpose as `paper/figures/FIGURE_PROVENANCE.md`: for every figure here, what
script made it, from which runs, with which seeds and config, so any number can be
traced and any figure regenerated.

Routing figures are **not yet cited by `paper.tex`**, so they live only in `docs/`.
When one is cited, copy it to `paper/figures/` and add a row there too.

---

## 1. Routing (CVRP) — Claim 1

### Environment

`envs/cvrp.py`, instance `default`: 10 customers + depot, demands sum 24 against
capacity 10, so the vehicle must reload 3 times. 37,918 states, **31,345 decision
states**, 22 observation features, 11 masked actions.

`travel_noise=0.15` adds zero-mean noise to leg costs. It leaves the optimal plan and
the exact oracle unchanged, but gives the rollout distributions something to vary
over. **It is required**: under determinism every pairwise total variation is 0 or 1
and the CCE score degenerates. This is the routing analogue of FrozenLake's slip.

### Training — where the checkpoints came from

```
python -m counterfactual_rl.agents.cvrp.train --seed <0..9> \
    --algorithm dqn-uniform --episodes 14000
```

Run dirs `c1full_s0` … `c1full_s9`, first dated 2026-08-06; seeds 3–9 were SLURM
job 274243. Key config (full block is in each run's `metrics.log` header):

| | |
|---|---|
| algorithm | `dqn-uniform` (plain DQN, uniform replay — the policy being scored, not a CCE arm) |
| instance / capacity | `default` / 10 |
| travel_noise | 0.15 |
| gamma | 0.99 |
| n_episodes | 14,000 (ε 1.0 → 0.05 over 7,000) |
| alpha / batch / buffer | 0.001 / 32 / 100,000 |
| target_update_freq | 200 |
| net | 2 × 64 |
| checkpoints | 14 per seed, every 500 episodes |

Sanity gate: plain DQN-uniform reaches **1.0000 of optimal** (random policy 0.62).
Exact optimal tour length 3.454742.

> **NOT FROZEN — but reproducible.** As of 2026-08-27 these runs exist only in the
> worktree `.claude/worktrees/cce-cvrp-logistics/src/counterfactual_rl/agents/cvrp/runs/`
> (9.8 GB, gitignored, not in `paper/repro/`). Losing them costs a rerun, not the
> result: training is **deterministic from the seed** — the same seed twice gives
> bit-identical weights (measured max abs diff 0.000e+00; a different seed differs by
> ~1.05, so the check is not vacuous) — and the config above is committed. From
> `timing.jsonl` the ten seeds took **8.1–9.0 min each**: ~84 min serial, ~9 min as a
> SLURM array.
>
> Caveat: that determinism check was run CPU-to-CPU. The original runs were on GPU, and
> float ops are not guaranteed bit-identical across different hardware. Same seed on the
> same device class should land exactly; a different GPU may shift the last digits.
>
> What is genuinely missing is convenience, not possibility. FrozenLake ships 2 MB of
> frozen Claim-1 checkpoints under `paper/repro/cache/checkpoints/`, so a reviewer runs
> `replot.py` and is done. Routing would make them retrain first. Freezing 3 stages x 10
> seeds is ~15 MB and would close the gap.

### Stage selection

`run_analysis.pick_checkpoints` takes `sorted(ckpt_*.pkl)` and picks
`[0]` / `[len//2]` / `[-1]`. With 14 checkpoints per seed that is:

| stage | file | episode |
|---|---|---|
| untrained | `ckpt_0001000.pkl` | 1,000 |
| mid | `ckpt_0008000.pkl` | 8,000 |
| trained | `ckpt_0014000.pkl` | 14,000 |

`trained` is the LAST checkpoint, deliberately not `best.pkl`: once the agent hits a
perfect score `best.pkl` can never improve on it (the update is a strict `>`), so it
freezes at whatever episode first reached the ceiling — which silently made `mid` and
`trained` the same weights and reported one measurement twice. A hash guard now fires
if any two stages are identical.

### Scoring — the generator

```
python -m counterfactual_rl.analysis.claim1.cvrp.run_analysis \
    --run-dir <runs>/c1full_s{0..9} \
    --aggregation mean --n-rollouts 25 --horizon 40 --travel-noise 0.15 \
    --gamma 0.99 --max-states 1000 --seed 42 \
    --out-dir docs/figures/real/claim1/cvrp_final_mean
```

Swap `--aggregation weighted_mean` and the out-dir for the other result set.

- **Oracle** (`claim1/cvrp/oracle.py`): exact backward induction over the full state
  space. `Oracle(s) = mean over legal a != a* of [Q*(s,a*) - Q*(s,a)]`. Validated
  against brute force on both TSP (all permutations) and CVRP (permutations ×
  optimal load-split).
- **CCE score** (`claim1/cvrp/score_states.py`): total variation among per-action
  return distributions from rollouts under the checkpoint's own greedy policy. It
  never sees the transition table, the reward function, or `Q*`.
- **Sample**: 1,000 of the 31,345 decision states, `numpy.default_rng(42)`.

### Result sets

| directory | seeds | aggregation | what it is |
|---|---|---|---|
| `cvrp_final_mean/` | 10 | `mean` | **HEADLINE.** ρ 0.576 → 0.765 → 0.741, p@10% 0.367 / 0.478 / 0.454 |
| `cvrp_final_weighted_mean/` | 10 | `max` (see §3) | same seeds under the rule that actually ran — ρ 0.536 → 0.693 → 0.667 |
| `cvrp/` | 3 | `max` | the FIRST routing Claim-1 result (2026-08-13, horizon 35). Superseded. Kept because the paper figure, the env spec table and the GIFs live here. **Its own JSON has no recovered provenance block — the sampling was never verified and it saved no `pairs_*.npz`, so treat its numbers as historical only.** |

### Figures

| file | generator | rebuild |
|---|---|---|
| `cvrp/fig_c1_paper_cvrp.png` | `claim1/cvrp/paper_figure.py` | from the two 10-seed dirs — see below |
| `fig_claim1_summary.png` | `claim1/claim1_summary.py` | reads routing from the JSON at import; FrozenLake is hardcoded (see §2) |
| `cvrp_final_*/fig_c1_scatter_cvrp.png` | `run_analysis.py` | needs the checkpoints |
| `cvrp_final_*/fig_c1_map_cvrp.png` | `run_analysis.py` | **BROKEN — do not use.** See §4 |
| `cvrp/*.gif` | `make_gif.py`, `explain_gif.py`, `teach_gif.py` | talks only, not paper assets |

```
python -m counterfactual_rl.analysis.claim1.cvrp.paper_figure \
    --mean-dir docs/figures/real/claim1/cvrp_final_mean \
    --max-dir  docs/figures/real/claim1/cvrp_final_weighted_mean \
    --out      docs/figures/real/claim1/cvrp

python -m counterfactual_rl.analysis.claim1.claim1_summary \
    --out docs/figures/real/claim1
```

Both rebuild from committed data alone — no runs needed. `fig_c1_paper_cvrp.png` is
verified to come back **byte-identical** from the committed `pairs_*.npz` and JSON.

### Re-checking a number without re-scoring

`cvrp_final_mean/pairs_{untrained,mid,trained}.npz` hold the raw per-state
`(oracle, cce)` pairs for seed 0. Every ρ and precision@k in the paper figure can be
recomputed from these in seconds:

```python
import numpy as np
from scipy.stats import spearmanr
z = np.load('docs/figures/real/claim1/cvrp_final_mean/pairs_mid.npz')
print(spearmanr(z['oracle'], z['cce']).statistic)   # -> 0.757 (seed 0, mid)
```

The oracle side is also a check on the whole pipeline: recomputing the oracle at
`gamma=0.99` and re-drawing the sample at `seed=42` reproduces `pairs_*.npz['oracle']`
exactly (sum 325.838389). That is how `gamma` and `sample_seed` were recovered on
2026-08-27 — `run_analysis.py` did not record them at the time. It does now.

---

## 2. FrozenLake — Claim 1 (already in the paper)

`frozen_lake/*.png` are the paper's published figures. **Not regenerated here.** They
are byte-identical to `paper/figures/`, and re-running the FL analysis overwrites
them — redirect the output if you run it.

The 2026-08-23 audit re-scored the published checkpoints and confirmed the paper:

```
stage        paper.tex        max (rerun)        mean (corrected)
untrained    0.319 +/- 0.114  0.319 +/- 0.114    0.326 +/- 0.105
mid          0.765 +/- 0.096  0.764 +/- 0.096    0.791 +/- 0.088
trained      0.889 +/- 0.031  0.888 +/- 0.031    0.895 +/- 0.032
```

Reproduce (the analysis writes no JSON, which is why `claim1_summary.py` hardcodes
these — with this command in its header):

```
python -m counterfactual_rl.analysis.claim1.frozen_lake.run_analysis \
    --aggregation <mean|weighted_mean> --ckpt-root paper/repro/cache/checkpoints
```

Full training provenance for those checkpoints is in
`paper/figures/FIGURE_PROVENANCE.md` §2.

---

## 3. Why the directory named `weighted_mean` holds `max` numbers

`analysis/metrics.py::compute_consequence_metric` honours `'weighted_mean'` only when
`action_probs` is supplied. FrozenLake and routing never supply it, so control falls
through to `max()`. Every FL and routing run to date has run `max` under a config that
says `weighted_mean`.

Behaviour is deliberately **unchanged** so `paper/repro/` keeps reproducing; the
fallback now raises a `RuntimeWarning`, and the Claim-1 scorers take an explicit
`aggregation` argument so an analysis can ask for `mean` without changing training.
Verified identical to master over 16,928 calls. Open issue #3.

---

## 4. Known-broken

`fig_c1_map_cvrp.png` — panels 2 and 3 render empty. `run_analysis.py:210` builds
`oracle` restricted to the 1,000 sampled states, then passes that restricted dict to
`plot_importance_map`, which asks for the ~10 states reached by the first leg out of
the depot. Those are essentially never in a random 1,000-of-31,345 draw, so
`if s2 in source` is never true and nothing is plotted. Fix: pass the full
`oracle_all`, and score those ~10 states explicitly for the CCE panel. Not yet done.
