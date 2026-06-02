#!/bin/bash
#SBATCH --job-name=cce_diagnostics
#SBATCH --output=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/diagnostics/logs/diag_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=manchadoa@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --account=undergrad_research
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=dh-node16,dh-node17,dh-node18

mkdir -p /home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/analysis/diagnostics/logs

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

# Same XLA flags as training: disable cuDNN autotuning (oversized grouped convs)
# and let rollout tensors grow on demand instead of pre-allocating.
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export PYTHONPATH="${PYTHONPATH}:/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src"

RUNS_ROOT="${DIAG_RUNS_ROOT:-/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src/counterfactual_rl/agents/shared/runs}"
RUN_IDS="${DIAG_RUN_IDS:-259281 259282 259283}"
CHUNKS="${DIAG_CHUNKS:-10 20 30 40 50 60 70 80 90 100}"
N_TRANS="${DIAG_N_TRANSITIONS:-1000}"
EPSILON="${DIAG_EPSILON:-0.05}"
MCTS_SIMS="${DIAG_MCTS_SIMS:-200}"
OUT_NPZ="${DIAG_OUT_NPZ:-/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/diagnostics/diagnostics.npz}"

echo "RUN_IDS:   $RUN_IDS"
echo "CHUNKS:    $CHUNKS"
echo "N_TRANS:   $N_TRANS   EPSILON: $EPSILON   MCTS_SIMS: $MCTS_SIMS"
echo "OUT_NPZ:   $OUT_NPZ"
echo ""

~/.conda/envs/counterfactual/bin/python -m counterfactual_rl.analysis.diagnostics.compute_diagnostics \
    --runs-root "$RUNS_ROOT" \
    --run-ids $RUN_IDS \
    --chunks $CHUNKS \
    --n-transitions "$N_TRANS" \
    --epsilon "$EPSILON" \
    --mcts-sims "$MCTS_SIMS" \
    --out "$OUT_NPZ"

echo "Diagnostics compute completed at $(date)"
