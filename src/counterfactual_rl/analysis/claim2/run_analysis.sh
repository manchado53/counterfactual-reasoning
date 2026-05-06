#!/bin/bash
#SBATCH --job-name=claim2_analysis
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/claim2/logs/analysis_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=dh-node16,dh-node17,dh-node18

mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/claim2/logs

export MPLBACKEND=Agg

# CUDA/NVIDIA library setup (mirrors training scripts)
PIP_LIBS=""
for nvidia_root in "$HOME/.conda/envs/counterfactual/lib/python3.12/site-packages/nvidia" \
                   "$HOME/.local/lib/python3.12/site-packages/nvidia"; do
    for pkg_lib in ${nvidia_root}/*/lib; do
        [ -d "$pkg_lib" ] && PIP_LIBS="${pkg_lib}:${PIP_LIBS}"
    done
done
SYS_LIBS=""
for p in /usr/local/cuda/lib64 /usr/local/cuda-12.5/lib64 /usr/local/cuda-12.0/lib64 \
         /usr/lib64/nvidia; do
    [ -d "$p" ] && SYS_LIBS="${SYS_LIBS:+${SYS_LIBS}:}${p}"
done
export LD_LIBRARY_PATH="${PIP_LIBS}${SYS_LIBS:+${SYS_LIBS}:}${LD_LIBRARY_PATH}"

export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

echo "ANALYSIS_MANIFEST:  $ANALYSIS_MANIFEST"
echo "ANALYSIS_ENV:       $ANALYSIS_ENV"
echo "ANALYSIS_THRESHOLD: $ANALYSIS_THRESHOLD"
echo "ANALYSIS_OUT:       $ANALYSIS_OUT"
echo ""

mkdir -p "$ANALYSIS_OUT"

~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.analysis.claim2.run_analysis \
    --manifest "$ANALYSIS_MANIFEST" \
    --env "$ANALYSIS_ENV" \
    --threshold "$ANALYSIS_THRESHOLD" \
    --out "$ANALYSIS_OUT"

echo "Analysis completed at $(date)"
