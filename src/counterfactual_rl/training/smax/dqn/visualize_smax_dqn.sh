#!/bin/bash
#SBATCH --job-name="SMAX DQN Visualize"
#SBATCH --output=logs/visualize_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Create directories
mkdir -p logs
mkdir -p videos

# Set matplotlib backend for headless
export MPLBACKEND=Agg

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

# Check/install ffmpeg (required for GIF generation)
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    conda install -y -c conda-forge ffmpeg
fi

echo "Starting visualization..."
echo "Checkpoint: models/smax_dqn_3m-1.pt"

# Run visualization
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.training.smax.dqn.visualize \
    --checkpoint models/smax_dqn_3m-1.pt \
    --output videos/gameplay.gif \
    --seed 42 \
    --episodes 3

echo "Visualization completed at $(date)"
