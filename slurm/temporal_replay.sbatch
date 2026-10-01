#!/bin/bash
set -euo pipefail
task_work="$1"
task_mode="$2"
task_python="${CCE_PYTHON:-$HOME/.conda/envs/counterfactual/bin/python}"
cd "$task_work/runtime"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_work/runtime/vendor:$task_work/runtime/src"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$task_python" "$task_work/runtime/scripts/train_temporal_replay.py" --work "$task_work" --mode "$task_mode"
