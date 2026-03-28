#!/bin/bash
#SBATCH --job-name=shuffled_resume
#SBATCH --account=ACCOUNT
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/shuffled_resume_%A_%a.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/shuffled_resume_%A_%a.err

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

source $PROJECT/ignition_index/venv/bin/activate

export HF_HOME=$PROJECT/ignition_index/HF_TOKEN_PLACEHOLDER
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME

# Only the 6 incomplete models
MODELS=(
    "gpt2-small"
    "gpt2-medium"
    "gpt2-xl"
    "pythia-410m"
    "pythia-1.4b"
    "pythia-6.9b"
)

MODEL_KEY=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "RESUMING shuffled-label controls for: $MODEL_KEY"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "=========================================="

cd $PROJECT/ignition_index/code

# Run with shuffled flag - will resume from partial checkpoint
python run_model.py \
    --model $MODEL_KEY \
    --shuffled \
    --signal_types S1

echo "=========================================="
echo "Completed: $MODEL_KEY"
echo "End time: $(date)"
echo "=========================================="
