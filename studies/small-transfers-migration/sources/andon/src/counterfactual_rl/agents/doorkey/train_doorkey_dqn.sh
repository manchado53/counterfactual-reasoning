#!/bin/bash
#SBATCH --job-name="DoorKey DQN"
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/andon-vending-bench-cce/src/counterfactual_rl/agents/doorkey/logs/train_%j.out
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
#SBATCH --nice=10000

WT=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/andon-vending-bench-cce

mkdir -p ${WT}/src/counterfactual_rl/agents/doorkey/logs

export MPLBACKEND=Agg

# CUDA/NVIDIA library setup (matches the FrozenLake / SMAX scripts)
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

# CRITICAL: prepend the WORKTREE src so the job loads DoorKey code (the editable install
# points at the MAIN repo, which has no doorkey package).
export PYTHONPATH="${WT}/src:${PYTHONPATH}"

if [ -n "$CONFIG_OVERRIDES_B64" ]; then
    echo "CONFIG_OVERRIDES: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"
else
    echo "CONFIG_OVERRIDES: none"
fi

~/.conda/envs/counterfactual/bin/python \
    -m counterfactual_rl.agents.doorkey.train

echo "Training completed at $(date)"
