#!/bin/bash
#SBATCH --job-name="SMAX DQN Training"
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/agents/smax/logs/train_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --exclude=dh-node16,dh-node17,dh-node18

# Create directories
mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/agents/smax/logs

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

# === GPU diagnostic ===
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
echo "GPU_DEVICE_ORDINAL=$GPU_DEVICE_ORDINAL"
nvidia-smi -L 2>&1 || echo "nvidia-smi not available"
echo "=== End GPU diagnostic ==="
echo ""

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Log config overrides for traceability (base64-encoded to survive SLURM --export)
if [ -n "$CONFIG_OVERRIDES_B64" ]; then
    echo "CONFIG_OVERRIDES: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"
else
    echo "CONFIG_OVERRIDES: none"
fi

# Run training (all parameters are in config.py)
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.agents.smax.train

echo "Training completed at $(date)"
