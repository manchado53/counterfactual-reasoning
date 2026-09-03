#!/bin/bash
# Regenerate every figure that reads the LIVE instrumentation sweep (array 274678).
# Safe to run mid-sweep: each module truncates all arms to the common episode floor,
# so a fast arm can never be compared against a slow one at different progress.
#
#   bash slurm/refresh_figures.sh            # default array 274678
#   bash slurm/refresh_figures.sh 274999     # some other array
set -u
WT=/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer
ARRAY=${1:-274678}
export PYTHONNOUSERSITE=1 PYTHONPATH=$WT/src MPLBACKEND=Agg JAX_PLATFORMS=cpu
PY=~/.conda/envs/counterfactual/bin/python
cd "$WT" || exit 1

fail=0
for m in jaxnav_instrument_figures jaxnav_score_distribution \
         jaxnav_score_ridgeline jaxnav_bootstrap_side_by_side \
         jaxnav_signal_funnel jaxnav_learning_curves; do
    printf '  %-34s ' "$m"
    if out=$($PY -m "counterfactual_rl.analysis.claim2.$m" --array "$ARRAY" 2>&1); then
        echo "OK   $(echo "$out" | grep -o 'floor:.*' | head -1)"
    else
        echo "FAILED"; echo "$out" | tail -5; fail=1
    fi
done

# Beside the figures, NOT /tmp -- /tmp is node-local here (lab-notebook gotcha) and
# the table is something you actually want to read later.
TABLE=docs/figures/real/claim2/jaxnav/prob_and_cost.txt
printf '  %-34s ' "jaxnav_prob_and_cost (old sweep)"
if $PY -m counterfactual_rl.analysis.claim2.jaxnav_prob_and_cost > "$TABLE" 2>&1; then
    echo "OK -> $TABLE"
else
    echo "FAILED"; tail -5 "$TABLE"; fail=1
fi

echo
echo "figures in docs/figures/real/claim2/jaxnav/"
exit $fail
