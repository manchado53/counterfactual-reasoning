#!/bin/bash
#SBATCH --job-name=dk12-10m-smoke
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0,5,10%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/doorkey12-10m-smoke-%A_%a.out

set -euo pipefail

root="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.codex/worktrees/doorkey-12x12-10m"
recipe="$root/experiments/doorkey/procedural-12x12-10m/training.json"
run_root="$root/experiments/doorkey/procedural-12x12-10m/runs"
output="$run_root/smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$root/src:$root"

mkdir -p "$run_root"
cd "$root"

exec /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python -m training.run train \
  --recipe "$recipe" --task "$SLURM_ARRAY_TASK_ID" --smoke --device gpu \
  --max-wall-seconds 840 --output "$output"
