# Graded-slip figures — provenance & reproduction

Experiment: **graded stochasticity sweep on FrozenLake 8×8** (GitHub issue #1,
07/30 meeting action item #1). Goal: test whether CCE+TD(mul)'s advantage over
DQN+PER grows as environment noise falls (Theorem 3) by filling the curve
between deterministic FL (a win) and full-slip FL (the null).

## The sweep

- Sweep def: `agents/frozen_lake/run_experiments.py` → `CLAIM2_GRADED_SLIP`
- 5 slip levels `p ∈ {0.0, 0.166, 0.333, 0.5, 0.666}` × 4 algorithms × 10 seeds = **200 runs**
- Algorithms: `DQN-Uniform`, `DQN+PER`, `DQN+CCE-only` (additive μ=1.0),
  `CCE+TD (mul)` (multiplicative). Additive-mixing cell dropped per 07/30 meeting.
- Config (matches the paper's FL runs): `map_name=8x8, n_episodes=15000, mu=0.25,
  consequence_metric=total_variation, epsilon_decay_episodes=7500, score_interval=100,
  cf_horizon=200, vectorized=True, early_stop_win_rate=0.95`.
- `slip_prob` knob added to `envs/frozen_lake.py` (outcome probs `[p/2, 1-p, p/2]`;
  p=0 ≡ deterministic, p=2/3 ≡ old `is_slippery=True`). Verified by `tests/test_frozen_lake_env.py`.
- Submitted 2026-08-03, all 200 completed. **14 runs died mid-training (cluster crashes)
  and are dropped** by `parse_logs.filter_complete_runs` → 186 used. The drop list is
  recorded in `graded_slip_summary.json` (`dropped_runs`).

## Data sources

| Artifact | Path |
|---|---|
| Frozen manifest (job_id→config) | `paper/repro/manifests/claim2_graded_slip_2026-08-03.json` |
| Parsed-array cache (rebuild w/o runs) | `paper/repro/cache/claim2_graded_slip.npz` (18 KB) |
| Cache builder (from raw runs) | `paper/repro/build_graded_slip_cache.py` |
| Raw runs (NOT committed; on the full disk) | `agents/frozen_lake/runs/<job_id>/metrics.log` |

## Figures (all from `analysis/claim2/graded_slip.py`)

| File | What it shows |
|---|---|
| `fig_iqm_vs_slip.png` | Final IQM win rate per algorithm vs slip (the deterministic cliff) |
| `fig_advantage_vs_slip.png` | **The direct test:** IQM(mul)−IQM(PER) + P(mul>PER) vs slip |
| `fig_advantage_vs_noise.png` | Same advantage vs outcome-entropy (true noise axis) |
| `fig_steps_to_threshold.png` | Speed: median env-steps to 0.5 / 0.9 win rate (the ceiling-proof view) |
| `fig_learning_curves_by_slip.png` | Paper-Fig-4-style IQM learning curves, one panel per slip level |
| `graded_slip_summary.json` | All numbers: final IQM, P(improve), steps-to-threshold, dropped runs |

## Reproduce

**Rebuild figures WITHOUT the raw runs (from the cache — always works):**
```
PYTHONPATH=src python -m counterfactual_rl.analysis.claim2.graded_slip \
    --from-cache paper/repro/cache/claim2_graded_slip.npz --out docs/figures/graded_slip
```

**Rebuild the cache from the raw runs (only while runs still exist):**
```
PYTHONPATH=src python paper/repro/build_graded_slip_cache.py
```

**Re-run the whole sweep from scratch (GPU):**
```
python -m counterfactual_rl.agents.frozen_lake.run_experiments claim2_graded_slip --dry-run
python -m counterfactual_rl.agents.frozen_lake.run_experiments claim2_graded_slip --max-concurrent 35
```

## Result (honest)

Theorem 3's smooth "advantage grows as slip falls" slope is **NOT** supported. The
CCE+TD(mul) win on final score is a **knife-edge at slip=0** (deterministic), flat-to-
slightly-negative at every slip>0. On the **speed** metric, mul reaches 0.5 win rate
fastest at most noise levels (a real early-learning edge the final-score ceiling hides).
Leading interpretation: FrozenLake's slip knob is **confounded** — more slip both
dilutes the CCE signal AND makes exploration easier (random walks reach the goal), and
the exploration effect dominates, producing the cliff. A clean test of Theorem 3 needs
outcome noise that does NOT ease exploration (e.g. reward noise). See the 2026-08-04
lab-notebook entry.
