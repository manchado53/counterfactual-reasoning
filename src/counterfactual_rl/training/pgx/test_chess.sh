#!/bin/bash
#SBATCH --job-name="Chess Test"
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/training/pgx/logs/test_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=dh-node16,dh-node17,dh-node18

mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/training/pgx/logs

cd /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/training/pgx

export MPLBACKEND=Agg

# CUDA/NVIDIA library setup (same as train_chess.sh)
PIP_LIBS=""
for nvidia_root in "$HOME/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia" \
                   "$HOME/.local/lib/python3.12/site-packages/nvidia"; do
    for pkg_lib in ${nvidia_root}/*/lib; do
        [ -d "$pkg_lib" ] && PIP_LIBS="${pkg_lib}:${PIP_LIBS}"
    done
done

SYS_LIBS=""
for p in /usr/local/cuda/lib64 /usr/local/cuda-12.5/lib64 /usr/local/cuda-12.0/lib64 \
         /usr/lib64/nvidia; do
    [ -d "$p" ] && SYS_LIBS="${SYS_LIBS:+${SYS_LIBS}:}${p}"
done

export LD_LIBRARY_PATH="${PIP_LIBS}${SYS_LIBS:+${SYS_LIBS}:}${LD_LIBRARY_PATH}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 || echo "nvidia-smi not available"
echo ""

export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# TEST_SCRIPT must be set via --export or environment
if [ -z "$TEST_SCRIPT" ]; then
    echo "ERROR: TEST_SCRIPT not set. Usage: TEST_SCRIPT=path/to/test.py sbatch test_chess.sh"
    exit 1
fi

echo "Running: $TEST_SCRIPT"
~/.conda/envs/counterfactual/bin/python "$TEST_SCRIPT"

echo "Test completed at $(date)"
