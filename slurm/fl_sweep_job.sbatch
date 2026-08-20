#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --account=undergrad_research
#SBATCH --job-name=fl_ess
#SBATCH --output=/home/ad.msoe.edu/manchadoa/fl_ess_logs/%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/fl_ess_logs/%j.err

# Worktree-local FrozenLake job. The repo's own train_frozen_lake_dqn.sh APPENDS the
# MAIN repo to PYTHONPATH, so a job launched from a worktree silently imports main-repo
# code and none of the worktree's changes -- here that would mean running without the
# ESS solver while the manifest claims target_ess_frac was set. This sets the worktree
# FIRST and sets PYTHONNOUSERSITE=1 to bypass the broken ~/.local site-packages.
# cpus-per-gpu is 8, not the repo default 32: at 32 only 2 of each node's 4 GPUs can be
# used, halving throughput for no benefit on FrozenLake.
WORKTREE=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKTREE/src
export MPLBACKEND=Agg
cd $WORKTREE

PIP_LIBS=""
for pkg_lib in "$HOME/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia"/*/lib; do
    [ -d "$pkg_lib" ] && PIP_LIBS="${pkg_lib}:${PIP_LIBS}"
done
for p in /usr/local/cuda/lib64 /usr/lib64/nvidia; do
    [ -d "$p" ] && PIP_LIBS="${PIP_LIBS}${p}:"
done
export LD_LIBRARY_PATH="${PIP_LIBS}${LD_LIBRARY_PATH}"

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "PYTHONPATH=$PYTHONPATH"
[ -n "$CONFIG_OVERRIDES_B64" ] && echo "CONFIG: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.agents.frozen_lake.train
