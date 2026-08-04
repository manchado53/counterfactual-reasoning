#!/bin/bash
#SBATCH --job-name="FrozenLake DQN"
#SBATCH --output=/home/ad.msoe.edu/manchadoa/graded_slip_logs/train_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --exclude=dh-node12,dh-node16,dh-node17,dh-node18
#SBATCH --nice=10000

mkdir -p /home/ad.msoe.edu/manchadoa/graded_slip_logs

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

# Worktree isolation: prepend THIS worktree's src so it wins over the editable
# install that points at the main repo (graded-slip experiment lives only here).
export PYTHONPATH="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/graded-slip-frozenlake/src:${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Log config overrides
if [ -n "$CONFIG_OVERRIDES_B64" ]; then
    echo "CONFIG_OVERRIDES: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"
else
    echo "CONFIG_OVERRIDES: none"
fi

# Provenance: prove which counterfactual_rl source tree is actually loaded.
~/.conda/envs/counterfactual/bin/python -c "import counterfactual_rl, os; print('CCRL_SOURCE =', os.path.dirname(counterfactual_rl.__file__))"

~/.conda/envs/counterfactual/bin/python \
    -m counterfactual_rl.agents.frozen_lake.train

echo "Training completed at $(date)"
