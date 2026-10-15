#!/bin/bash
set -euo pipefail

: "${CCE_REPO:?Set CCE_REPO to the worktree under test}"
: "${CCE_PYTHON:?Set CCE_PYTHON to the project Python executable}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) recipe="experiments/smax/historical/training.json"; task=3; label="smax" ;;
  1) recipe="experiments/chess/historical/training.json"; task=3; label="chess" ;;
  2) recipe="experiments/cvrp/historical/training.json"; task=3; label="cvrp" ;;
  3) recipe="experiments/rpdqn/historical/training.json"; task=3; label="rpdqn" ;;
  4) recipe="experiments/cvrp_budget/budget/training.json"; task=0; label="cvrp-budget" ;;
  5) recipe="experiments/cvrp_budget/budget/training.json"; task=4; label="cvrp-option-a" ;;
  *) echo "Unknown smoke task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${CCE_REPO}/src:${CCE_REPO}"

cd "${CCE_REPO}"
nvidia-smi --query-gpu=name,driver_version --format=csv

output="${CCE_REPO}/runs/shared-learning-smoke-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${label}"
exec "${CCE_PYTHON}" -m training.run train \
  --recipe "${recipe}" \
  --task "${task}" \
  --output "${output}" \
  --device gpu \
  --smoke \
  --max-wall-seconds 600
