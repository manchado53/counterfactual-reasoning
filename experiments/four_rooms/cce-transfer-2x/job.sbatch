#!/bin/bash
#SBATCH --job-name=four-rooms-2x
#SBATCH --partition=batch
#SBATCH --account=undergrad_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/four-rooms-2x-%A_%a.out
set -euo pipefail
: "${FR_ROOT:?}" "${FR_MODE:?}" "${FR_COMMIT:?}" "${FR_PYTHON:?}"
cd "$FR_ROOT"
test "$(git rev-parse HEAD)" = "$FR_COMMIT"
git diff --exit-code HEAD -- src training experiments/four_rooms
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$FR_ROOT/src:$FR_ROOT"
export PATH="$(dirname "$FR_PYTHON")/../lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
args=()
limit=86000
if [[ "$FR_MODE" == smoke ]]; then args+=(--smoke); limit=900; fi
nvidia-smi --query-gpu=name,driver_version --format=csv
exec "$FR_PYTHON" -m training.run train \
  --recipe experiments/four_rooms/cce-transfer-2x/training.json \
  --task "$SLURM_ARRAY_TASK_ID" --device gpu --max-wall-seconds "$limit" \
  --output "runs/four-rooms-2x/$FR_MODE/task_$SLURM_ARRAY_TASK_ID" "${args[@]}"
