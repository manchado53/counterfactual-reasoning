# Sparse Four Rooms: CCE transfer

Question: does full CCE improve learning over Uniform and standard rank PER?
This is a new study. A successful smoke is not a learning result.

## Source and deviations

Sutton, Precup & Singh (1999), *Between MDPs and semi-MDPs*,
[paper, Figure 2 and section 3.1](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf),
[DOI](https://doi.org/10.1016/S0004-3702(99)00052-1).
The 13-by-13 grid has 104 open cells, including four doors. The right horizontal
wall is one row below the left wall. Coordinates are zero-based, from top left:
doors (3,6), (6,2), (7,9), (10,6). Start (1,1); goal (9,9), the figure's G2 cell.
Movement is intended direction with probability 2/3, each other direction 1/9.
Walls leave position unchanged. Actions are up, down, left, right (0,1,2,3).

**Our Sparse Four Rooms variant:** +1 on first goal arrival, zero elsewhere;
end at goal or step 200. These reward/start/cap choices are study rules, not a
claim about the original benchmark. No options or shaping. We use gamma .99,
not the paper's .9 rooms example. Fully observed state contains the cell and
elapsed time: one-hot cell plus elapsed/200. Timeout is part of this finite
horizon task, with no value bootstrap after timeout. Goal wins a step-200 tie.

## Frozen protocol

15 tasks: Uniform 0–4, rank PER 5–9, CCE 10–14; training seeds 0–4.
All use the same shared `DiscreteAgent`, network (128 x 3 ReLU), Double DQN,
Adam .0003, gradient norm cap 10, gamma .99, batch 128, capacity 100000,
one update per 16 new steps after 1024 warm-up steps, target sync every 125
updates (2000 new steps), 32 parallel environments x 32 steps per block.
Budget: 512 blocks = 524288 steps = 32704 updates. Epsilon falls linearly
from 1 to .05 over 262144 steps, then stays fixed; no early stop.

Full CCE transfers the fixed collector/no-decay multiplicative recipe from
`experiments/jaxnav/ranked-td-decay/training.json` (task 13): Wasserstein,
weighted alternatives, horizon 60, one rollout/action, greedy continuation,
gamma .99, target-network max-value bootstrap at horizon, action softmax
 temperature .5, score every insertion, percentile rank normalization,
exponents mu_c=mu_delta=1, eps=.01, priority exponent .25. No CCE decay.
The observed JaxNav lead was mid-training, not a claim of a final win.
The simulator is branched exactly; stochastic returns are sampled, **not an
exact expected-return calculation**. All four actions are rolled out from the
saved pre-transition state. The real action selects the reference; the other
three are alternatives. One frozen Q table per collection block is exactly the
network evaluated at all finite states, not a learned dynamics model.

Uniform and rank PER share the same storage, insertion, update and sampler code.
Only replay probabilities vary. Standard PER uses p proportional to descending
rank(|TD|)^(-.25), with average ranks for ties. This differs from CCE's ascending
percentile rank mix and from the historical raw-TD JaxNav PER baseline.
Importance weights are 1/(N*p) for all arms (Uniform gives ones), matching shared
CCE code. New entries inherit the running maximum TD magnitude.
CCE scoring consumes its own RNG; it cannot change collection draws.
Transfer choices: finite-state encoding, four actions, time-aware cap, smaller
collection blocks, fixed step budget/epsilon clock, standard inverse-rank PER.
These choices are frozen before looking at learning curves.

## Evaluation and uncertainty

One bank: 1000 fixed-start episodes, each with 200 explicit step keys from
JAX `random.split(PRNGKey(20261015), 1000*200)`. Every seed/arm/checkpoint uses
identical starts and keys; evaluation never advances training or replay RNGs.
Save bank bytes and SHA256. Evaluate at step zero and every 16384 steps.
Save per-episode success and lengths. Successful length is undefined (null)
when no episode succeeds; never substitute 200 or zero.

Thick mean, dashed median, faint seeds, descriptive SEM band. Curves use only
observed points. Summary final/AUC/peak uses the common observed checkpoints
of all 15 runs and states its horizon. Final-budget labels require completion.
AUC is trapezoidal success area divided by the common horizon. Peak is each
seed's largest observed success (selection-biased, descriptive).
Probability curves use the exact paired seed bootstrap: all 5^5 resamples of
five paired seed differences. Report fraction with positive mean difference,
not a Bayesian probability that CCE is truly better. Also report seed win
fraction. Five seeds give limited precision. No forward fill or extrapolation.

## Reproduce

Use the existing JAX 0.9.1 runtime, or the pinned dependency files in
`requirements/migration-cpu.lock` and `requirements/migration-cuda12.lock`.
See `docs/migration-install.md` for known historical dependency conflicts.
From this checkout, with the desired Python on PATH:

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src:$PWD"
python -m pytest tests/test_four_rooms.py tests/test_four_rooms_report.py -q
python -m training.run validate --recipe experiments/four_rooms/cce-transfer/training.json --task 10
python -m training.run train --recipe experiments/four_rooms/cce-transfer/training.json --task 10 --smoke --device cpu --max-wall-seconds 300 --output runs/four-rooms/cpu-smoke
python -m training.run train --recipe experiments/four_rooms/cce-transfer/training.json --task 10 --device gpu --max-wall-seconds 86000 --output runs/four-rooms/cce-seed0
python -m analysis.studies.four_rooms --runs runs/four-rooms/full --out runs/four-rooms/report
```

`training.json` contains every exact setting. Each attempt saves source ZIP,
source hashes, package versions, device, resolved settings and status.
Raw files: eval.jsonl, progress.jsonl, evaluation_bank.npz, eval_STEP.npz,
replay_STEP.npz (raw CCE scores, TD magnitudes and sampling probabilities),
last.pkl. A checkpoint is not an exact resumable training snapshot.

SLURM uses `job.sbatch`; dry-run and smoke gates precede full submission.
A single full array has concurrency 7, leaving one job slot for the CPU monitor.
Use shared allocations; never cancel unrelated jobs. Monitor events are limited
to startup/failure, 25/50/75/100% budget milestones, and completion.

### Launch record (2026-10-15)

Base: origin/master 5b3a608 (user approved the remote default; origin/main absent).
Core commit: 5ea837b. Report/launcher commit: 1563b27.
CPU smoke: all three arms completed. GPU smoke array 284071 tasks 0,5,10:
all COMPLETED, exit 0; actual device cuda:0 / Tesla T4, driver 595.71.05.
Each GPU smoke collected 384 transitions and ran 16 updates. CCE scored 384/384.
Full training results are separate and are not established by these checks.

Exact cluster launch from this checkout (fresh output directories required):

```bash
export FR_ROOT="$PWD" FR_MODE=full FR_COMMIT="$(git rev-parse HEAD)"
export FR_PYTHON=/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python
mkdir -p logs
sbatch --test-only --array=0-14%7 experiments/four_rooms/cce-transfer/job.sbatch
sbatch --parsable --array=0-14%7 experiments/four_rooms/cce-transfer/job.sbatch
# Use the returned array ID below. Run detached under a durable host session.
python -m training.monitor_four_rooms --root "$PWD" --job ARRAY_ID
```

The monitor writes `runs/four-rooms/monitor/events.jsonl`, updates the report at
quarter-budget milestones and completion, and saves its PID/status. Events are
local logs; this script does not send email or chat messages.
