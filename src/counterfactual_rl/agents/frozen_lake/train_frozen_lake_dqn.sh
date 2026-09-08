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
#SBATCH --time=14:00:00
# dh-node12 is a CONFIRMED BAD NODE (lab-notebook 2026-08-13) and the JaxNav array
# driver has always excluded it; this script never did. On 2026-09-08 that cost 12 of
# 36 coverage_dose runs: 12/14 jobs landing on dh-node12 were SIGKILLed within 1-13
# minutes, while all 22 jobs on every other node completed.
#SBATCH --exclude=dh-node12,dh-node16,dh-node17,dh-node18
#SBATCH --nice=10000

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

# REPO_SRC is exported by run_experiments.py and points at the src/ of whichever
# checkout submitted the job. It MUST be prepended, not appended: the editable install
# resolves to the MAIN repo, so a job submitted from a git worktree silently trains
# main-repo code otherwise -- which is exactly what happened on 2026-09-08, when 12
# FrozenLake jobs ran without the buffer instrumentation that was the point of the run.
# SLURM executes a spool copy of this script, so $0/BASH_SOURCE cannot be used to
# discover the checkout; it has to be passed in. Falls back to the main repo so any
# existing caller that does not set it behaves as before.
: "${REPO_SRC:=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src}"
export PYTHONPATH="${REPO_SRC}:${PYTHONPATH}"
# ~/.local carries a broken cffi; keep it off sys.path.
export PYTHONNOUSERSITE=1
echo "REPO_SRC=$REPO_SRC"
echo "PYTHONPATH=$PYTHONPATH"

# Log config overrides
if [ -n "$CONFIG_OVERRIDES_B64" ]; then
    echo "CONFIG_OVERRIDES: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"
else
    echo "CONFIG_OVERRIDES: none"
fi

~/.conda/envs/counterfactual/bin/python \
    -m counterfactual_rl.agents.frozen_lake.train

echo "Training completed at $(date)"
