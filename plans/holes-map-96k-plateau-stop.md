# Holes map, 96k episodes, with real plateau early-stopping

We're about to double the holes-map episode budget again (48k → 96k). Last time we found both
CCE and PER had mostly flattened by 48k — running longer blind risks burning GPU hours on a
run that's already converged. The user wants an early-stop this time, but the only early-stop
we have is threshold-based ("stop once win_rate >= X"), and that doesn't fit: CCE and PER land
at very different final levels here (CCE anywhere 13-76%, PER 58-71%), so there's no single
success threshold that means "done" for both. What we actually need is a plateau detector —
stop when the metric has stopped improving, regardless of what level it plateaued at.

## Design: a smoothed, patience-based plateau stopper

A single raw eval (100 test episodes) swings ±10-15 points between checkpoints just from noise
(we saw this directly in the 48k run's trend numbers), so triggering off one bad eval would
stop runs prematurely. Instead: track a **smoothed** win-rate (mean of the last few evals), keep
the best smoothed value seen so far, and stop once several eval checkpoints pass without a real
improvement over that best.

New shared utility — `src/counterfactual_rl/agents/shared/early_stopping.py`:
```python
class PlateauEarlyStopper:
    def __init__(self, patience, min_delta, smooth_window, min_episodes):
        ...
    def update(self, episode, win_rate) -> bool:
        # returns True when training should stop
```
Reused by both trainers, same pattern as the existing shared `buffers.py`/`timing.py` — avoids
duplicating stateful window logic in two near-identical `learn()` loops.

New config knobs in `agents/jax_nav/config.py` (defaults):
- `early_stop_patience: 20` (eval checkpoints with no improvement → stop; at `eval_interval=250`
  that's 5000 episodes of no progress)
- `early_stop_min_delta: 0.02` (2 points of smoothed win-rate counts as "still improving")
- `early_stop_smooth_window: 5` (average the last 5 evals before comparing)
- `early_stop_min_episodes`: set per-run to `epsilon_decay_episodes` (don't declare a plateau
  while the policy is still mostly random — no point stopping during exploration)

Wire into both `learn()` loops — `agents/jax_nav/dqn_vectorized.py` (~line 254, next to the
existing `early_stop_win_rate` check) and `agents/jax_nav/consequence_dqn_vectorized.py` (~line
214, identical spot) — as an additional `or` condition alongside the existing threshold check
(kept, in case a future run does have a real target). Print which condition triggered.

## The run itself

`slurm/sweep_holes_long_v3.py`, mirroring `sweep_holes_long_v2.py`: same holes-map config (8×8,
10% obstacles, `coll_rew=0`, `score_interval=500`), `n_episodes=96000`,
`epsilon_decay_episodes=40000` (keeps the same ~42% decay ratio used in the 24k and 48k runs),
5 seeds each for CCE-mul and PER (10 jobs total, matching the last run so it's a clean
continuation, not a new experiment shape). SLURM `--time` bumped to account for runs that don't
early-stop (worst case ~2x the 48k run's longest job, ~2h40m — round up to `--time=03:30:00`).

**Not submitting anything until this plan is approved** — per the user's explicit ask to
confirm before jobs go out.

## Verification
- Unit-check `PlateauEarlyStopper` standalone (a few hand-fed win-rate sequences: monotonic
  climb never stops, flat-noise-around-a-value stops around the expected checkpoint, a late
  jump after a plateau resets the counter) before wiring into the trainers.
- Short local smoke run (`--episodes 3000 --override early_stop_patience=3
  --override early_stop_smooth_window=2`) confirming a run actually exits early instead of
  running to completion, before submitting the real 96k jobs.
