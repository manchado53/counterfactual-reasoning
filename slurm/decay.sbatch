#!/bin/bash
set -euo pipefail
task_work="$1"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cd "$task_work"
nvidia-smi --query-gpu=name,driver_version --format=csv
exec /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python "$task_work/scripts/decay_run.py" "$task_work"
