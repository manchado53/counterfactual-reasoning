#!/bin/bash
#SBATCH --job-name=cce_convergence
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/diagnostics/logs/conv_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=dh-node16,dh-node17,dh-node18

mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/diagnostics/logs

export MPLBACKEND=Agg

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

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

RUNS_ROOT="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/agents/shared/runs"
OUT_NPZ="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/diagnostics/convergence.npz"

echo "Convergence check starting at $(date)"
~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.analysis.diagnostics.convergence_check \
    --runs-root "$RUNS_ROOT" \
    --run-id 259281 --chunk 100 \
    --n-states 300 \
    --epsilon 0.05 \
    --budgets 50 200 800 3200 \
    --out "$OUT_NPZ"
echo "Convergence check completed at $(date)"
