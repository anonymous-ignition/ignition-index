#!/bin/bash
#SBATCH --job-name=shuffled_missing
#SBATCH --account=ACCOUNT
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/shuffled_missing_%A_%a.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/shuffled_missing_%A_%a.err

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

source $PROJECT/ignition_index/venv/bin/activate

export HF_HOME=$PROJECT/ignition_index/HF_TOKEN_PLACEHOLDER
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME

# The 5 missing models from first run
MODELS=(
    "gemma2-2b"
    "gemma2-9b"
    "huginn-3.5b"
    "mamba-1.4b"
    "mamba-2.8b"
)

MODEL_KEY=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "T1.1 shuffled-label for: $MODEL_KEY"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "=========================================="

cd $PROJECT/ignition_index/code

# Run with shuffled flag
python run_model.py \
    --model $MODEL_KEY \
    --shuffled \
    --signal_types S1

echo "=========================================="
echo "Completed: $MODEL_KEY"
echo "End time: $(date)"
echo "=========================================="
