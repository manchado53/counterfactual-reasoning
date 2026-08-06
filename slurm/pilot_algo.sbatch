#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=undergrad_research
#SBATCH --job-name=jaxnav_cmp
#SBATCH --output=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.err

# Go/no-go pilot: CCE-mul vs PER on a stabilized (Double-DQN, replay-ratio 8) sparse
# JaxNav task. Pass ALGO / MIXING / SEED via --export.

WORKTREE=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKTREE/src
cd $WORKTREE

ALGO=${ALGO:-dqn}
EXTRA=""
if [ "$ALGO" = "consequence-dqn" ]; then EXTRA="--mixing ${MIXING:-multiplicative}"; fi
echo "host=$(hostname)  ALGO=$ALGO $EXTRA  SEED=${SEED:-0}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.agents.jax_nav.train \
  --algorithm $ALGO $EXTRA --episodes 8000 --seed ${SEED:-0} \
  --override sparse_reward=True --override 'map_size=(6,6)' --override fill=0.0 \
  --override goal_radius=1.0 --override max_steps=150 --override epsilon_decay_episodes=4000 \
  --override eval_interval=200 --override eval_episodes=100 \
  --override n_envs=256 --override collect_steps=32
