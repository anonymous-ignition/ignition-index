#!/bin/bash
#SBATCH --job-name=t2_1_huginn_s1
#SBATCH --account=ACCOUNT
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/t2_1_huginn_%j.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/t2_1_huginn_%j.err

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

source $PROJECT/ignition_index/venv/bin/activate

export HF_HOME=$PROJECT/ignition_index/HF_TOKEN_PLACEHOLDER
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME
export HF_TOKEN=YOUR_TOKEN_HERE

echo "=========================================="
echo "T2.1: HUGINN ITERATION-AXIS PROBING"
echo "Signal type: S1 only (paper-ready version)"
echo "Start time: $(date)"
echo "=========================================="

cd $PROJECT/ignition_index/code

python run_huginn_iteration.py --signal_types S1

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Huginn iteration probing complete"
    echo "Output: huginn_iteration_probing.pkl"
else
    echo "FAILED: Exit code $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
