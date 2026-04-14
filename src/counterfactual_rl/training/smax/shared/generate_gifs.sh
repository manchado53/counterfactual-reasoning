#!/bin/bash
#SBATCH --job-name="Generate GIFs"
#SBATCH --output=logs/gifs_%j.out
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
# Generate gameplay GIFs from a trained model checkpoint.
#
# Usage:
#   sbatch generate_gifs.sh <checkpoint_path> [episodes] [output_dir]
#
# Examples:
#   sbatch generate_gifs.sh runs/223139/best.pkl
#   sbatch generate_gifs.sh runs/228198/best.pkl 5
#   sbatch generate_gifs.sh runs/228198/best.pkl 3 my_gifs/

set -e

mkdir -p logs

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [episodes] [output_dir]}"
EPISODES="${2:-3}"
OUTPUT_DIR="${3:-$(dirname "$CHECKPOINT")/gifs}"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"
export MPLBACKEND=Agg

PYTHON=~/.conda/envs/counterfactual/bin/python

echo "Checkpoint: $CHECKPOINT"
echo "Episodes:   $EPISODES"
echo "Output dir: $OUTPUT_DIR"

$PYTHON -m counterfactual_rl.training.smax.shared.visualize \
  --checkpoint "$CHECKPOINT" \
  --episodes "$EPISODES" \
  --output "$OUTPUT_DIR/gameplay.gif"

echo "GIFs saved to $OUTPUT_DIR"
