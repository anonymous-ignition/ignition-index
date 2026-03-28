#!/bin/bash
# ════════════════════════════════════════════════════════════════
# jobs/run_training_dynamics.sh
# H4: Pythia 410M — 154 training checkpoints + PELT changepoint.
# Single job, ~14 hours.
#
# Submit:   sbatch jobs/run_training_dynamics.sh
# Can run in parallel with run_array.sh.
# ════════════════════════════════════════════════════════════════

#SBATCH --job-name=ign_h4_dynamics
#SBATCH --account=ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/training_dynamics_%j.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/training_dynamics_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=AUTHOR@mail.ubc.ca

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

echo "H4 training dynamics start: $(date)"
python run_training_dynamics.py
echo "H4 done: $(date)"
