#!/bin/bash
#SBATCH --job-name=cce_suitability
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=undergrad_research
#SBATCH --output=/home/ad.msoe.edu/manchadoa/suitability_logs/%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/suitability_logs/%j.err

set -e
cd /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning
PY=~/.conda/envs/counterfactual/bin/python

# Matched full runs: det-trained consequence-dqn-mul in det env, slippery in slippery env.
$PY -m counterfactual_rl.analysis.suitability.run_full \
  --det-run   src/counterfactual_rl/agents/frozen_lake/runs/257440 \
  --stoch-run src/counterfactual_rl/agents/frozen_lake/runs/255545 \
  --out           docs/figures/suitability/scorecard.json \
  --fig           docs/figures/suitability/scorecard.png \
  --dashboard-in  docs/figures/mock_preview/dashboard.html \
  --dashboard-out docs/figures/suitability/dashboard_real.html \
  --cf-n-rollouts 60 --visit-episodes 100 --eval-episodes 50 \
  --horizons 10 25 50 100 200 --horizon-states 16

echo "SUITABILITY_FULL_DONE"
