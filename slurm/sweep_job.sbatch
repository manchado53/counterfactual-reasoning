#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=undergrad_research
#SBATCH --job-name=jaxnav_sweep
#SBATCH --output=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.err

# Generic sweep job: the full config arrives via CONFIG_OVERRIDES_B64 (base64 JSON),
# which train.py merges over DEFAULT_CONFIG.

WORKTREE=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKTREE/src
cd $WORKTREE
echo "host=$(hostname)  job=$SLURM_JOB_ID"
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.agents.jax_nav.train
