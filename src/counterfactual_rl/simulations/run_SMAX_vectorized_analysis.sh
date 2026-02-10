#!/bin/bash
#SBATCH --job-name="SMAX Vectorized CF"
#SBATCH --output=logs/job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8

# Create logs directory if it doesn't exist
mkdir -p logs

# Use non-interactive matplotlib backend for headless rendering on compute nodes
export MPLBACKEND=Agg

# Run the vectorized simulation
~/.conda/envs/counterfactual/bin/python smax_vectorized_random_policy.py

echo "Job completed at $(date)"
