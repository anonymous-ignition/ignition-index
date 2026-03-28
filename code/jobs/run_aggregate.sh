#!/bin/bash
# ════════════════════════════════════════════════════════════════
# jobs/run_aggregate.sh
# Collects all model .pkl files, runs H1-H5 stats, builds
# figures and Table 1. CPU-only job (~2 hours).
#
# Submit AFTER all run_array jobs are confirmed complete:
#   sbatch jobs/run_aggregate.sh
#
# Or run interactively:
#   salloc --mem=32G --cpus-per-task=16 --time=2:00:00 \
#          --account=ACCOUNT
#   source /home/AUTHOR/projects/ACCOUNT/AUTHOR/ignition_index/venv/bin/activate
#   cd /home/AUTHOR/projects/ACCOUNT/AUTHOR/ignition_index/code
#   python aggregate.py --signal_types S1 S2 S3
# ════════════════════════════════════════════════════════════════

#SBATCH --job-name=ign_aggregate
#SBATCH --account=ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/AUTHOR/scratch/ignition_index/logs/aggregate_%j.out
#SBATCH --error=/home/AUTHOR/scratch/ignition_index/logs/aggregate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=AUTHOR@mail.ubc.ca

module purge
module load python/3.11

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

source $PROJECT/ignition_index/venv/bin/activate

export HF_HOME=$PROJECT/ignition_index/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export SCRATCH=$SCRATCH
export PROJECT=$PROJECT

cd $PROJECT/ignition_index/code

echo "Aggregation start: $(date)"
python aggregate.py --signal_types S1 S2 S3

# Copy final outputs to project space for permanent storage
# (scratch is purged after 60 days — project is permanent)
mkdir -p $PROJECT/ignition_index/final_outputs
cp -r $SCRATCH/ignition_index/figures $PROJECT/ignition_index/final_outputs/
cp -r $SCRATCH/ignition_index/tables  $PROJECT/ignition_index/final_outputs/
cp -r $SCRATCH/ignition_index/results $PROJECT/ignition_index/final_outputs/
echo "Outputs copied to $PROJECT/ignition_index/final_outputs/"
echo "Aggregation done: $(date)"
