#!/bin/bash
#SBATCH --job-name=ign_mamba_1p4b
#SBATCH --account=ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/mamba_1p4b_%j.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/mamba_1p4b_%j.err

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
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd $PROJECT/ignition_index/code

echo "mamba-1.4b S2+S3 Start: $(date)"
python run_model.py --model "mamba-1.4b" --signal_types S2 S3
echo "mamba-1.4b S2+S3 Done: $(date)"
