# Benchmark-style reward batch — goal +4 / crash -4 / time -0.01

## Why

Our JaxNav reward lets an agent wander for free. Under crash = -1 that produced a timid agent:
uniform's timeouts went 29% -> 57% and win rate 59% -> 35%, because "stay safe and do nothing"
scores 0 while trying scores negative early in training.

The published JaxNav setup (No Regrets, NeurIPS 2024; `sfl/train/config/env/jaxnav.yaml`) never
allows that. It uses goal +4, crash -4, a -0.01 per-step time cost, plus dense distance shaping and
a wall-proximity term. We take the **sparse half only**: symmetric goal/crash and the time cost. That
removes free wandering without the dense shaping, which would change what CCE measures ("got closer"
instead of "the choice mattered").

## What runs

One new reward cell, 8x8 fill 0.3, 250k episodes, 5 seeds each:

- `uniform_bench` — plain replay
- `per_bench` — TD priorities
- `cov_wm_bench` — greedy_cov_wm exactly (16% coverage, greedy R=1, H=60 + bootstrap,
  wasserstein, weighted_mean), only the reward differs

Compared only inside this cell. ~87 GPU-h, ~15 h wall.

## Code

`dt_rew` is currently forced to 0 whenever `sparse_reward=True`. Add a `dt_rew` knob that defaults
to None (off), so every existing run stays identical, and pass it through from the trainer config.
Pin the analysis baselines on `goal_rew` and `dt_rew` as well as `coll_rew`, or these runs leak into
the default uniform/PER lines — the fourth time that bug shape would bite.
