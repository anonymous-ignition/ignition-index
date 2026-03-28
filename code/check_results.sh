#!/bin/bash
# ════════════════════════════════════════════════════════════════
# check_results.sh
# Quick status check: which models are done, which are pending.
# Run any time on the login node.
#
# Usage:
#   bash check_results.sh
# ════════════════════════════════════════════════════════════════

SCRATCH=/home/AUTHOR/scratch
RESULTS=$SCRATCH/ignition_index/results
LOGS=$SCRATCH/ignition_index/logs

MODELS=(
    "gpt2-small"    "gpt2-medium"   "gpt2-xl"
    "pythia-70m"    "pythia-410m"   "pythia-1.4b"   "pythia-6.9b"
    "gemma2-2b"     "gemma2-9b"
    "huginn-3.5b"
    "mamba-1.4b"    "mamba-2.8b"
)

echo "================================================================"
echo "  Ignition Index — Results Status"
echo "  Results dir: $RESULTS"
echo "================================================================"
DONE=0; PENDING=0
for mk in "${MODELS[@]}"; do
    safe="${mk//-/_}"
    safe="${safe//./_}"
    pkl="$RESULTS/model_${safe}.pkl"
    if [ -f "$pkl" ]; then
        size=$(du -sh "$pkl" 2>/dev/null | cut -f1)
        echo "  [DONE]    $mk ($size)"
        ((DONE++))
    else
        echo "  [PENDING] $mk"
        ((PENDING++))
    fi
done
echo "----------------------------------------------------------------"
echo "  Done: $DONE / $((DONE+PENDING))"
echo ""

echo "Active SLURM jobs:"
squeue -u AUTHOR --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || \
    echo "  (squeue not available)"
echo ""

echo "Recent log tails:"
for log in $(ls -t $LOGS/*.log 2>/dev/null | head -3); do
    echo "--- $(basename $log) ---"
    tail -3 "$log" 2>/dev/null
done
