#!/bin/bash
#SBATCH --job-name=value-score-gate
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-1%2
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORMS JAX_PLATFORM_NAME
python_bin=/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  "$python_bin" experiments/value-score-replay/run.py --task 0 --smoke
  "$python_bin" experiments/value-score-replay/run.py --task 1 --smoke
else
  "$python_bin" experiments/value-score-replay/run.py --task 6 --smoke
  "$python_bin" experiments/value-score-replay/run.py --task 7 --smoke
fi
