#!/bin/bash
# ════════════════════════════════════════════════════════════════
# jobs/run_array.sh
# Full experiment array: all 12 models x S1+S2+S3.
# Each array task = 1 model on 1 A100 GPU, running independently.
#
# Submit:   sbatch jobs/run_array.sh
# Monitor:  squeue -u AUTHOR
#           tail -f /home/AUTHOR/scratch/ignition_index/logs/gpt2-small.log
# ════════════════════════════════════════════════════════════════

#SBATCH --job-name=ignition_idx
#SBATCH --account=ACCOUNT
#SBATCH --array=0-11
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/array_%A_%a.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/array_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=AUTHOR@mail.ubc.ca

# ── Model list (index matches --array=0-11) ───────────────────────
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
    "huginn-3.5b"
    "mamba-1.4b"
    "mamba-2.8b"
)
MODEL=${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID → model: $MODEL"

# ── Environment ───────────────────────────────────────────────────
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

# ── Run ───────────────────────────────────────────────────────────
echo "[$MODEL] Start: $(date)"
python run_model.py \
    --model "$MODEL" \
    --signal_types S1 S2 S3 \
    
echo "[$MODEL] Done: $(date)"
