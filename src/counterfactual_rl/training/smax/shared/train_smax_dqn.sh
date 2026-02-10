#!/bin/bash
#SBATCH --job-name="SMAX DQN Training"
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Create directories
mkdir -p logs
mkdir -p models

# Set matplotlib backend for headless
export MPLBACKEND=Agg

# CUDA/NVIDIA library setup
# NVIDIA pip packages are split across conda env AND ~/.local (user pip install).
# Scan both locations for lib dirs.
PIP_LIBS=""
for nvidia_root in "$HOME/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia" \
                   "$HOME/.local/lib/python3.12/site-packages/nvidia"; do
    for pkg_lib in ${nvidia_root}/*/lib; do
        [ -d "$pkg_lib" ] && PIP_LIBS="${pkg_lib}:${PIP_LIBS}"
    done
done

# System CUDA toolkit (low priority, appended after pip)
SYS_LIBS=""
for p in /usr/local/cuda/lib64 /usr/local/cuda-12.5/lib64 /usr/local/cuda-12.0/lib64 \
         /usr/lib64/nvidia; do
    [ -d "$p" ] && SYS_LIBS="${SYS_LIBS:+${SYS_LIBS}:}${p}"
done

export LD_LIBRARY_PATH="${PIP_LIBS}${SYS_LIBS:+${SYS_LIBS}:}${LD_LIBRARY_PATH}"

# === Diagnostic (remove after confirming it works) ===
echo "LD_LIBRARY_PATH (first 3 entries):"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -5
echo ""
~/.conda/envs/counterfactual/bin/python -c "
import ctypes, os, sys
# Try loading the critical libraries directly
for lib_name in ['libcuda.so.1', 'libcudart.so.12', 'libcudnn.so.9', 'libcublasLt.so.12', 'libnvrtc.so.12']:
    try:
        ctypes.CDLL(lib_name)
        print(f'  OK: {lib_name}')
    except OSError as e:
        print(f'  FAIL: {lib_name} -> {e}')
" 2>&1
echo "=== End diagnostic ==="
echo ""

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Default backend (override with: sbatch --export=BACKEND=pytorch train_smax_dqn.sh)
BACKEND=${BACKEND:-jax}

# Run training
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.training.smax.shared.train \
    --backend $BACKEND \
    --scenario 3m \
    --n-episodes 20000 \
    --obs-type world_state \
    --eval-episodes 100

echo "Training completed at $(date)"
