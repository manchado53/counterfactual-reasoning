#!/bin/bash
#SBATCH --job-name=value-score-t4
#SBATCH --partition=batch
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=1,3,5,7,9,11%6
#SBATCH --no-requeue
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORMS JAX_PLATFORM_NAME
case "$SLURM_ARRAY_TASK_ID" in 1|3|5|7|9|11) ;; *) exit 2 ;; esac
python_bin=/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python
if [ "${2:-}" = smoke ]; then
    exec "$python_bin" experiments/value-score-replay/run.py --task "$SLURM_ARRAY_TASK_ID" --smoke
fi
exec "$python_bin" experiments/value-score-replay/run.py --task "$SLURM_ARRAY_TASK_ID"
