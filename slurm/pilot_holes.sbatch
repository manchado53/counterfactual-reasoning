#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=undergrad_research
#SBATCH --job-name=jaxnav_holes
#SBATCH --output=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.err

# Option A "holes test": CCE-mul vs PER on an OBSTACLE map (fill=0.1, crash = 0-reward
# terminal = a FrozenLake-style hole), with tuned scoring (score_interval=500 -> wall-clock-fair).

WORKTREE=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKTREE/src
cd $WORKTREE

ALGO=${ALGO:-dqn}
EXTRA=""
if [ "$ALGO" = "consequence-dqn" ]; then EXTRA="--mixing multiplicative --override score_interval=500"; fi
echo "host=$(hostname)  ALGO=$ALGO $EXTRA  SEED=${SEED:-0}"

~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.agents.jax_nav.train \
  --algorithm $ALGO $EXTRA --episodes 12000 --seed ${SEED:-0} \
  --override sparse_reward=True --override 'map_size=(8,8)' --override fill=0.1 \
  --override goal_radius=0.8 --override coll_rew=0.0 --override max_steps=200 \
  --override epsilon_decay_episodes=5000 --override eval_interval=250 --override eval_episodes=100 \
  --override n_envs=256 --override collect_steps=32
