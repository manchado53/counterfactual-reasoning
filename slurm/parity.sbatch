#!/bin/bash
set -euo pipefail
suite="$1"
python_bin="$2"
gpu_type="${3:-h100}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$suite/candidate/src:$suite/candidate"
cd "$suite/candidate"
nvidia-smi --query-gpu=name,driver_version --format=csv
exec "$python_bin" -m experiments.parity.training --reference "$suite/reference" \
  --output "$suite/results" --profile full-size --device gpu --gpu-type "$gpu_type"
