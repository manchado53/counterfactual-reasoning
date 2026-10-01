#!/bin/bash
set -euo pipefail
task_work="$1"
task_mode="$2"
task_python="${CCE_PYTHON:-$HOME/.conda/envs/counterfactual/bin/python}"
task_variant=$("$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])]["variant"])' "$task_work/manifest.json" "$SLURM_ARRAY_TASK_ID")
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_work/vendor:$task_work/$task_variant/src"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
cd "$task_work"
nvidia-smi --query-gpu=name,driver_version --format=csv
"$task_python" "$task_work/scripts/train_carry_pilot.py" --work "$task_work" --mode "$task_mode"
