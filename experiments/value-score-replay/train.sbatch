#!/bin/bash
#SBATCH --job-name=value-score-pair
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:45:00
#SBATCH --array=0-11%2
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORMS JAX_PLATFORM_NAME
exec /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python experiments/value-score-replay/run.py --task "$SLURM_ARRAY_TASK_ID"
