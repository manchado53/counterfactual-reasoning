#!/bin/bash
#SBATCH --job-name=dk16-t4-smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:t4:1
#SBATCH --array=0,5,10%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/doorkey16-t4-smoke-%A_%a.out

set -euo pipefail

root="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.codex/worktrees/doorkey-paper-parity"
recipe="$root/experiments/doorkey/fullstate-16x16/training.json"
run_root="$root/experiments/doorkey/fullstate-16x16/runs"
output="$run_root/t4_smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$root/src:$root"
export PATH="/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"

mkdir -p "$run_root"
cd "$root"

exec /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python -m training.run train \
  --recipe "$recipe" --task "$SLURM_ARRAY_TASK_ID" --smoke --device gpu \
  --max-wall-seconds 1140 --output "$output"
