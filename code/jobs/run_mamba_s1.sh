#!/bin/bash
#SBATCH --job-name=ign_mamba_s1
#SBATCH --account=ACCOUNT
#SBATCH --array=0-1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/mamba_s1_%A_%a.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/mamba_s1_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=AUTHOR@mail.ubc.ca

MODEL_LIST=(
    "mamba-1.4b"
    "mamba-2.8b"
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
    --signal_types S1
echo "[$MODEL] Done: $(date)"
