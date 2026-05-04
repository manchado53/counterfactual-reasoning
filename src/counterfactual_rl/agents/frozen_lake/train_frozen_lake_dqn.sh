#!/bin/bash
#SBATCH --job-name="FrozenLake DQN"
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/agents/frozen_lake/logs/train_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=dh-node16,dh-node17,dh-node18

mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/agents/frozen_lake/logs

export MPLBACKEND=Agg

# CUDA/NVIDIA library setup (matches train_smax_dqn.sh)
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

export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Log config overrides
if [ -n "$CONFIG_OVERRIDES_B64" ]; then
    echo "CONFIG_OVERRIDES: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"
else
    echo "CONFIG_OVERRIDES: none"
fi

# Algorithm is passed via env vars (defaults shown)
ALGORITHM="${ALGORITHM:-consequence-dqn}"
MAP_NAME="${MAP_NAME:-4x4}"
MIXING="${MIXING:-additive}"
MU="${MU:-}"

echo "Algorithm: $ALGORITHM  Map: $MAP_NAME  Mixing: $MIXING  MU: ${MU:-default}"

MU_ARG=""
[ -n "$MU" ] && MU_ARG="--mu $MU"

~/.conda/envs/counterfactual/bin/python \
    -m counterfactual_rl.agents.frozen_lake.train \
    --algorithm "$ALGORITHM" \
    --map "$MAP_NAME" \
    --mixing "$MIXING" \
    $MU_ARG

echo "Training completed at $(date)"
