#!/bin/bash
#SBATCH --job-name=cce-control-t4
#SBATCH --account=undergrad_research
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0,2,4%3
#SBATCH --no-requeue
set -euo pipefail
cd "$1"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORMS JAX_PLATFORM_NAME
case "$SLURM_ARRAY_TASK_ID" in 0|2|4|6|8|10) ;; *) exit 2 ;; esac
python_bin=/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python
if [ "${2:-}" = smoke ]; then
    exec "$python_bin" experiments/value-score-replay/run.py --task "$SLURM_ARRAY_TASK_ID" --smoke
fi
exec "$python_bin" experiments/value-score-replay/run.py --task "$SLURM_ARRAY_TASK_ID"
