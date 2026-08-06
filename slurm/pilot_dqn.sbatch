#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=undergrad_research
#SBATCH --job-name=jaxnav_pilot
#SBATCH --output=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/jaxnav_logs/%j.err

# Stage-1 gate: does plain DQN+PER learn on JaxNav? EASY config (obstacle-free,
# small map, big goal, dense guidance) + goal-reach metric = clean "trainer works?" test.

WORKTREE=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer
export PYTHONNOUSERSITE=1                 # dodge broken ~/.local cffi (crashes import jaxmarl)
export PYTHONPATH=$WORKTREE/src           # load THIS worktree's code, not the main editable install
cd $WORKTREE

echo "host=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.agents.jax_nav.train \
  --algorithm dqn --episodes 8000 \
  --override sparse_reward=False --override 'map_size=(6,6)' --override fill=0.0 \
  --override goal_radius=1.0 --override max_steps=150 --override epsilon_decay_episodes=3000 \
  --override eval_interval=200 --override eval_episodes=100 \
  --override n_envs=256 --override collect_steps=32 --override early_stop_win_rate=0.9
