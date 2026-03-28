#!/bin/bash
# ════════════════════════════════════════════════════════════════
# setup_env.sh
# Run ONCE on the Narval login node to create the virtual
# environment and install all dependencies.
#
# Usage:
#   bash setup_env.sh
# ════════════════════════════════════════════════════════════════

set -e   # exit immediately on any error

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

echo "=== Setting up Ignition Index environment ==="
echo "PROJECT: $PROJECT"
echo "SCRATCH: $SCRATCH"

# ── Load modules ─────────────────────────────────────────────────
module purge
module load python/3.11 cuda/12.2 cudnn/8.9

# ── Create venv ──────────────────────────────────────────────────
cd $PROJECT/ignition_index/
if [ -d "venv" ]; then
    echo "venv already exists — skipping creation"
else
    echo "Creating virtual environment..."
    python -m venv venv
fi
source venv/bin/activate

# ── Set HF cache to project space ────────────────────────────────
export HF_HOME=$PROJECT/ignition_index/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

# ── Install dependencies ─────────────────────────────────────────
echo "Installing Python packages..."
pip install --upgrade pip --quiet

pip install torch==2.10.0 --quiet
pip install transformers==4.57.6 --quiet
pip install transformer-lens==2.17.0 --quiet
pip install sae-lens==6.27.1 --quiet
pip install scikit-learn==1.8.0 scipy==1.17.1 ruptures==1.1.10 --quiet
pip install datasets conllu seaborn pandas tqdm matplotlib accelerate --quiet
pip install spacy --quiet

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# ── Mamba SSM (compiles CUDA — ~5 min) ───────────────────────────
echo "Installing mamba-ssm (compiles CUDA, takes ~5 min)..."
pip install ninja --quiet
pip install mamba-ssm==2.3.0 --no-build-isolation || \
    echo "WARNING: mamba-ssm failed — Mamba models will be skipped"

echo ""
echo "=== Environment setup complete ==="
echo "Test with: python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'"
