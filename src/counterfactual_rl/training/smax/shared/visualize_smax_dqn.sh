#!/bin/bash
#SBATCH --job-name="SMAX DQN Visualize"
#SBATCH --output=logs/visualize_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Create directories
mkdir -p logs
mkdir -p videos

# Set matplotlib backend for headless
export MPLBACKEND=Agg

# CUDA/NVIDIA library setup — scan both conda env and ~/.local for nvidia libs
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

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Default backend (override with: sbatch --export=BACKEND=pytorch visualize_smax_dqn.sh)
BACKEND=${BACKEND:-jax}

# Default checkpoint (override with: sbatch --export=CHECKPOINT=path/to/model visualize_smax_dqn.sh)
CHECKPOINT=${CHECKPOINT:-models/smax_dqn.pkl}

# Check/install ffmpeg (required for GIF generation)
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    conda install -y -c conda-forge ffmpeg
fi

echo "Starting visualization..."
echo "Backend: $BACKEND"
echo "Checkpoint: $CHECKPOINT"

# Run visualization
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.training.smax.shared.visualize \
    --backend $BACKEND \
    --checkpoint $CHECKPOINT \
    --output videos/gameplay.gif \
    --seed 42 \
    --episodes 3

echo "Visualization completed at $(date)"
