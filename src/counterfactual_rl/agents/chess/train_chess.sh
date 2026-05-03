#!/bin/bash
#SBATCH --job-name="Chess DQN Training"
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/training/pgx/logs/train_%j.out
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

# Create log directory
mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/training/pgx/logs

cd /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/training/pgx

# Set matplotlib backend for headless
export MPLBACKEND=Agg

# CUDA/NVIDIA library setup
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

# === GPU diagnostic ===
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 || echo "nvidia-smi not available"
echo "=== End GPU diagnostic ==="
echo ""

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Log config overrides
if [ -n "$CONFIG_OVERRIDES_B64" ]; then
    echo "CONFIG_OVERRIDES: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"
else
    echo "CONFIG_OVERRIDES: none"
fi

# Run training
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.training.pgx.dqn_jax.train

echo "Training completed at $(date)"
