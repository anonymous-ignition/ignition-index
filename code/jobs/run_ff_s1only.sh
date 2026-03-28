#!/bin/bash
# ════════════════════════════════════════════════════════════════
# jobs/run_ff_s1only.sh
# Phase 1 verification: 9 feedforward models, S1 signal only.
# Run this first to verify the full pipeline works before
# committing to the full S1+S2+S3 run.
#
# Submit:   sbatch jobs/run_ff_s1only.sh
# Monitor:  squeue -u AUTHOR
# ════════════════════════════════════════════════════════════════

#SBATCH --job-name=ign_ff_s1
#SBATCH --account=ACCOUNT
#SBATCH --array=0-8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/ff_s1_%A_%a.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/ff_s1_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=AUTHOR@mail.ubc.ca

MODEL_LIST=(
    "gpt2-small"
    "gpt2-medium"
    "gpt2-xl"
    "pythia-70m"
    "pythia-410m"
    "pythia-1.4b"
    "pythia-6.9b"
    "gemma2-2b"
    "gemma2-9b"
)
MODEL=${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}

module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

source $PROJECT/ignition_index/venv/bin/activate

export HF_HOME=$PROJECT/ignition_index/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export SCRATCH=$SCRATCH
export PROJECT=$PROJECT

cd $PROJECT/ignition_index/code

echo "[$MODEL] Start: $(date)"
python run_model.py \
    --model "$MODEL" \
    --signal_types S1 \
    
echo "[$MODEL] Done: $(date)"
