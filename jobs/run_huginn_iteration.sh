#!/bin/bash
#SBATCH --job-name=huginn_iteration
#SBATCH --account=ACCOUNT
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/huginn_iteration_%j.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/huginn_iteration_%j.err

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

source $PROJECT/ignition_index/venv/bin/activate

export HF_HOME=$PROJECT/ignition_index/HF_TOKEN_PLACEHOLDER
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME

echo "=========================================="
echo "T2.1: Huginn Iteration-Axis Probing"
echo "Fixed depth: layer 16"
echo "Probing across iterations 1-64"
echo "Start time: $(date)"
echo "=========================================="

cd $PROJECT/ignition_index/code

# Run iteration probing (all signal types for completeness)
python run_huginn_iteration.py --signal_types S1 S2 S3

echo "=========================================="
echo "Completed Huginn iteration probing"
echo "Output: huginn_iteration_probing.pkl"
echo "End time: $(date)"
echo "=========================================="
