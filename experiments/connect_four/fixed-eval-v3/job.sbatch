#!/bin/bash
#SBATCH --job-name=c4-fixed-eval-v3
#SBATCH --partition=dgxh100
#SBATCH --account=undergrad_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/c4-fixed-eval-v3-%A_%a.out
set -euo pipefail
: "${C4_ROOT:?}" "${C4_MODE:?}" "${C4_COMMIT:?}"
cd "$C4_ROOT"
test "$(git rev-parse HEAD)" = "$C4_COMMIT"
git diff --exit-code HEAD -- src training experiments/connect_four/fixed-eval-v3
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$C4_ROOT/.runtime:$C4_ROOT/src:$C4_ROOT"
export PATH="/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
run_root="$C4_ROOT/experiments/connect_four/fixed-eval-v3/runs"
mkdir -p "$run_root"
args=()
limit=172000
if [[ "$C4_MODE" == smoke ]]; then
    args+=(--smoke)
    limit=3500
fi
exec /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python -m training.run train \
    --recipe experiments/connect_four/fixed-eval-v3/training.json \
    --task "$SLURM_ARRAY_TASK_ID" --device gpu --max-wall-seconds "$limit" \
    --output "$run_root/${C4_MODE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "${args[@]}"
