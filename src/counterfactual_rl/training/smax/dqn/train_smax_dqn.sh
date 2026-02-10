#!/bin/bash
#SBATCH --job-name="SMAX DQN Training"
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Create directories
mkdir -p logs
mkdir -p models

# Set matplotlib backend for headless
export MPLBACKEND=Agg

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Run training
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.training.smax.dqn.train \
    --scenario 3m \
    --n-episodes 40000 \
    --save-path models/smax_dqn_3m_concatenated.pt \
    --plot-path training_curves_concatenated.png \
    --obs-type concatenated

echo "Training completed at $(date)"
    