#!/bin/bash
#SBATCH --job-name=dk-paper-gate
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=23:55:00
#SBATCH --output=logs/doorkey-paper-gate-%j.out

set -euo pipefail

root="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.codex/worktrees/doorkey-paper-parity"
recipe="$root/experiments/doorkey/vlmrb-paper-parity/training.json"
run_root="$root/experiments/doorkey/vlmrb-paper-parity/runs"
output="$run_root/gate_${SLURM_JOB_ID}_0"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$root/src:$root"
export PATH="/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"

mkdir -p "$run_root"
cd "$root"

exec /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python -m training.run train \
  --recipe "$recipe" --task 0 --device gpu --max-wall-seconds 85500 --output "$output"
