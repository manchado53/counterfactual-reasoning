#!/bin/bash
#SBATCH --job-name="Chess C1 s2"
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/claim1/chess/logs/chess_claim1_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --exclude=dh-node16,dh-node17,dh-node18

mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/claim1/chess/logs

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

PIP_LIBS=""
for nvidia_root in "$HOME/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia"                    "$HOME/.local/lib/python3.12/site-packages/nvidia"; do
    for pkg_lib in ${nvidia_root}/*/lib; do
        [ -d "$pkg_lib" ] && PIP_LIBS="${pkg_lib}:${PIP_LIBS}"
    done
done

SYS_LIBS=""
for p in /usr/local/cuda/lib64 /usr/local/cuda-12.5/lib64 /usr/local/cuda-12.0/lib64          /usr/lib64/nvidia; do
    [ -d "$p" ] && SYS_LIBS="${SYS_LIBS:+${SYS_LIBS}:}${p}"
done

export LD_LIBRARY_PATH="${PIP_LIBS}${SYS_LIBS:+${SYS_LIBS}:}${LD_LIBRARY_PATH}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 || echo "nvidia-smi not available"

export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

~/.conda/envs/counterfactual/bin/python \
    -m counterfactual_rl.analysis.claim1.chess.run_analysis \
    --n-episodes 100 \
    --n-rollouts 32 \
    --horizon 20 \
    --top-k 10 \
    --chunk-size 32 \
    --seed 2 \
    --cache-path src/counterfactual_rl/analysis/claim1/chess/records_cache_100g_s2.pkl

echo "Job completed at $(date)"
