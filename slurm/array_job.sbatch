#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --account=undergrad_research
#SBATCH --job-name=jaxnav_fact
#SBATCH --output=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%A_%a.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%A_%a.err

# Array variant of sweep_job.sbatch. Each task reads its own base64 config from
# line (SLURM_ARRAY_TASK_ID + 1) of the configs file passed in CONFIG_FILE.
#
# An array is used rather than 180 individual sbatch calls so the concurrency
# cap (--array=0-179%20) is enforced by SLURM itself. The repo's
# slurm_throttle.py blocks in the SUBMITTING process, which would mean keeping a
# shell alive for the ~94h this sweep takes; the cap would vanish the moment
# that process died.
#
# PYTHONPATH puts the WORKTREE FIRST on purpose: the repo's own launcher scripts
# APPEND the main repo, so a job launched from a worktree silently imports
# main-repo code. PYTHONNOUSERSITE=1 bypasses the broken ~/.local site-packages.

WORKTREE=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKTREE/src
export MPLBACKEND=Agg
cd $WORKTREE

if [ -z "$CONFIG_FILE" ]; then
    echo "FATAL: CONFIG_FILE not set" >&2
    exit 1
fi

LINE=$((SLURM_ARRAY_TASK_ID + 1))
CONFIG_OVERRIDES_B64=$(sed -n "${LINE}p" "$CONFIG_FILE")
if [ -z "$CONFIG_OVERRIDES_B64" ]; then
    echo "FATAL: no config on line $LINE of $CONFIG_FILE" >&2
    exit 1
fi
export CONFIG_OVERRIDES_B64

echo "host=$(hostname)  array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONFIG: $(echo "$CONFIG_OVERRIDES_B64" | base64 -d)"

~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.agents.jax_nav.train
