#!/usr/bin/env bash
set -euo pipefail

: "${CCE_SOURCE_ROOT:?Set CCE_SOURCE_ROOT to the immutable checkout}"
: "${CCE_RUN_ROOT:?Set CCE_RUN_ROOT to a fresh output root}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${CCE_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

recipe="${CCE_SOURCE_ROOT}/experiments/cartpole/per-positive-control/training.json"
task="${SLURM_ARRAY_TASK_ID}"
output="${CCE_RUN_ROOT}/task-${task}"
args=(train --recipe "${recipe}" --task "${task}" --output "${output}"
      --device cpu --max-wall-seconds "${CCE_MAX_WALL_SECONDS:-7000}")
if [[ "${CCE_SMOKE:-0}" == "1" ]]; then
  args+=(--smoke)
fi
cd "${CCE_SOURCE_ROOT}"
exec "${CCE_PYTHON:-/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python}" +  -m training.run "${args[@]}"
