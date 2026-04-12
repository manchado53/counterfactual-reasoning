# Bootstrap Sample Efficiency Analysis

Post-hoc statistical analysis to measure whether consequence-DQN is more **sample efficient** than PER and uniform DQN — meaning: does it reach better performance with fewer training episodes?

## What it does

Reads completed `metrics.log` files from finished training runs and produces a 3-panel figure per metric showing:

1. **Learning curves** — all individual seed runs + mean per algorithm
2. **Bootstrap distributions** — win rate distributions at a chosen checkpoint
3. **P(consequence > baseline)** — probability that consequence-DQN beats each baseline at every checkpoint, with 90% confidence intervals

## How to run

```bash
# From the shared/ directory
python evals/bootstrap_analysis.py experiments/full_algorithm_comparison_2026-04-07.json

# With options
python evals/bootstrap_analysis.py experiments/full_algorithm_comparison_2026-04-07.json \
    --scenario 3m \
    --focus-checkpoint 10000 \
    --metrics win_rate avg_return
```

Produces one PNG per metric, saved next to the manifest file:
```
experiments/full_algorithm_comparison_2026-04-07_3m_win_rate_bootstrap.png
experiments/full_algorithm_comparison_2026-04-07_3m_avg_return_bootstrap.png
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--focus-checkpoint` | `10000` | Episode to zoom in on in Panel 2 |
| `--n-bootstrap` | `10000` | Inner bootstrap iterations (higher = more precise P estimate) |
| `--n-outer` | `200` | Outer iterations for CI on P (controls band width) |
| `--scenario` | all | Filter to one scenario (e.g. `3m`, `8m`) |
| `--metrics` | both | `win_rate`, `avg_return`, or both |
| `--seed` | `42` | RNG seed for reproducibility |

## How the bootstrap works

### Inner bootstrap — P(A > B)

At a given checkpoint, each algorithm has N win rates (one per seed):

```
Consequence: [75%, 73%, 67%]   ← N=3 seeds
PER:         [68%, 59%, 69%]
```

The inner bootstrap resamples each with replacement 10,000 times and counts how often consequence's mean exceeds PER's mean:

```
P(consequence > PER) = 7,200 / 10,000 = 0.72
```

### Outer bootstrap — CI on P

With only N=3 seeds, the P estimate itself is uncertain. The outer bootstrap resamples the seeds 200 times, recomputes P(A>B) each time, then takes the 5th–95th percentile:

```
90% CI: [0.55, 0.88]
```

This is what the shaded band in Panel 3 represents. Wide bands = you need more seeds.

## What the panels mean

### Panel 1 — Learning curves
- Faint lines: individual seeds. Shows actual inter-seed variance.
- Solid line: mean across seeds.
- Dashed vertical line: the focus checkpoint shown in Panel 2.

### Panel 2 — Bootstrap distributions at focus checkpoint
- Each histogram is the distribution of 10,000 bootstrap means at that checkpoint.
- Overlap between histograms = ambiguity about which method is better.
- Annotated with P(consequence > each baseline).

### Panel 3 — P over training
- Above 0.5 = consequence-DQN is more likely to be ahead.
- Rising above 0.5 **early in training** = sample efficiency evidence.
- Shaded band = 90% CI. Wide bands = not enough seeds to be confident.

## How many seeds do you need?

| Seeds | Typical CI width | What you can claim |
|-------|-----------------|-------------------|
| 3 | ~±0.30 | Trend only — visual evidence |
| 10 | ~±0.15 | Defensible probabilistic claim |
| 20+ | ~±0.08 | Publishable statistical evidence |

The script prints a warning if any group has fewer than 10 seeds.

## What it reads

Each run's `runs/{job_id}/metrics.log` — written during training at every `eval_interval` episodes (default: every 100 episodes). No changes to training code are needed.

## What it reuses

Parsing utilities imported directly from `summarize_experiment.py`:
- `parse_metrics_log` — reads metrics.log into row dicts
- `load_manifests` — merges manifest JSON files
- `config_label` — human-readable algorithm names
- `config_key` — groups runs by config, excluding seed
