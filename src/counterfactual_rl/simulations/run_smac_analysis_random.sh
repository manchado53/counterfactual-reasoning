#!/bin/bash
#SBATCH --job-name="Smac Counterfactual Analysis"
#SBATCH --output=logs/job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Load any required modules (adjust based on your cluster)
# module load python/3.8

# Run the simulation
 ~/.conda/envs/counterfactual/bin/python smac_random_policy.py

echo "Job completed at $(date)"
