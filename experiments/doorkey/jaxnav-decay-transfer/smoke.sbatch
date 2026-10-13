#!/bin/bash
#SBATCH --job-name=dk-xfer-smoke
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0,5,10,15,20,25%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/doorkey-transfer-smoke-%A_%a.out

set -euo pipefail

root="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.codex/worktrees/doorkey-jaxnav-decay-transfer"
recipe="$root/experiments/doorkey/jaxnav-decay-transfer/training.json"
run_root="$root/experiments/doorkey/jaxnav-decay-transfer/runs"
output="$run_root/smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$root/src:$root"

mkdir -p "$run_root"
cd "$root"

exec /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python -m training.run train \
  --recipe "$recipe" \
  --task "$SLURM_ARRAY_TASK_ID" \
  --smoke \
  --device gpu \
  --max-wall-seconds 540 \
  --output "$output"
