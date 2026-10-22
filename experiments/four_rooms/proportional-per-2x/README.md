# One proportional PER control — 2× Sparse Four Rooms

One full-budget run, seed 0, task 0. This is a matched control for the existing
2× study, not a new five-seed claim. Existing jobs and source worktrees remain
untouched. Use the same 404-cell map, fixed start/goal, movement noise, sparse
reward and 200-step cap. Source/deviations: [2× recipe](../cce-transfer-2x/README.md).

Only replay selection changes. Sampling probabilities are proportional to
`(|TD error| + .01)^.25`, using the shared replay equation with raw magnitudes,
no rank normalization, and zero CCE contribution. This matches the existing
shared proportional PER implementation's probability formula. The .25 priority
exponent and full `1/(N*p)` importance correction stay equal to the existing
study; no new alpha/beta schedule is introduced. In the repo's parameter names,
`PER_parameters.beta` is the priority exponent (often called alpha in PER
literature); it is not the importance-correction exponent.

All other settings are frozen: 128×3 hidden layers, Double DQN, Adam .0003,
batch 128, capacity 100000, warm-up 1024, one update per 16 steps, target sync
2000 steps, epsilon 1→.05 over 262144 steps, 524288 total steps/32704 updates,
and the same 1000-episode evaluation key bank. The shared sampler/storage and
TD update are unchanged. No counterfactual rollout or CCE scoring runs in this
arm; CCE-only config values remain inactive for parity with the parent recipe.

New outputs live under `runs/four-rooms-proportional-per`. The single-seed report
compares only seed 0 of all four arms, using exact shared evaluation checkpoints.
No SEM or probability-of-superiority claim is valid from this one seed.
The monitor refreshes that report on quarter-budget milestones and completion.

## Reproduce

Use the pinned JAX 0.9.1 runtime described in the parent study. From the checkout:

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src:$PWD"
python -m pytest tests/test_four_rooms_proportional.py -q
python -m training.run validate --recipe experiments/four_rooms/proportional-per-2x/training.json --task 0
python -m training.run train --recipe experiments/four_rooms/proportional-per-2x/training.json --task 0 --smoke --device cpu --max-wall-seconds 300 --output runs/four-rooms-proportional-per/cpu/task_0
python -m training.run train --recipe experiments/four_rooms/proportional-per-2x/training.json --task 0 --device gpu --max-wall-seconds 86000 --output runs/four-rooms-proportional-per/full/task_0
```

GPU smoke and full launch use `job.sbatch` with `--array=0%1`. Both require a
SLURM dry run. Full launch requires passing tests and CPU/T4 smoke. This adds at
most one GPU job to the parent study's cap of seven, for at most eight Four
Rooms jobs total. No unrelated jobs are stopped or reconfigured.

```bash
export FR_ROOT="$PWD" FR_MODE=full FR_COMMIT="$(git rev-parse HEAD)"
export FR_PYTHON=/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python
mkdir -p logs
sbatch --test-only --array=0%1 experiments/four_rooms/proportional-per-2x/job.sbatch
sbatch --parsable --array=0%1 experiments/four_rooms/proportional-per-2x/job.sbatch
```

Raw eval/progress/replay files, config, source bundle, device and evaluation bank
identity follow the parent study format. A successful smoke establishes runnable
code, not a learning result.

Matched single-seed plot and quiet monitor (replace the reference path with your
saved parent-study full-run directory and ARRAY_ID with the submitted job):

```bash
python -m analysis.studies.four_rooms_single --control runs/four-rooms-proportional-per/full --reference ../four-rooms-2x/runs/four-rooms-2x/full --out runs/four-rooms-proportional-per/report
python -m training.monitor_four_rooms --root "$PWD" --job ARRAY_ID --study-dir runs/four-rooms-proportional-per --tasks 1 --reference-runs ../four-rooms-2x/runs/four-rooms-2x/full
```

Validation: 48 shared/study tests passed, then all four proportional-specific
tests passed after the report test was added (49 unique tests). CPU smoke and
T4 GPU smoke array 284123 completed: 384 transitions, 16 updates, zero CCE
scorings. `validation.json` records devices and bank/source hashes.
