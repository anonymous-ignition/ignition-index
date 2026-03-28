#!/bin/bash
# ════════════════════════════════════════════════════════════════
# download_models.sh
# Pre-downloads all HuggingFace models to project space.
# Run ONCE on the login node BEFORE submitting any jobs.
# Compute nodes have limited/no internet — models must be cached.
#
# Usage:
#   bash download_models.sh
#   bash download_models.sh --gemma    # include Gemma 2 (needs HF token)
# ════════════════════════════════════════════════════════════════

PROJECT=/home/AUTHOR/projects/ACCOUNT/AUTHOR
SCRATCH=/home/AUTHOR/scratch

module purge
module load python/3.11 cuda/12.2 cudnn/8.9
source $PROJECT/ignition_index/venv/bin/activate

export HF_HOME=$PROJECT/ignition_index/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

INCLUDE_GEMMA=false
for arg in "$@"; do
    [ "$arg" = "--gemma" ] && INCLUDE_GEMMA=true
done

echo "=== Downloading models to $HF_HOME ==="
echo "This may take 30-60 minutes depending on connection speed."
echo ""

python3 - <<PYEOF
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Standard models (no token required)
MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-xl",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-6.9b",
    "tomg-group-umd/huginn-0125",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf",
]

include_gemma = os.environ.get("INCLUDE_GEMMA", "false") == "true"
if include_gemma:
    MODELS += ["google/gemma-2-2b", "google/gemma-2-9b"]

for mid in MODELS:
    print(f"Downloading {mid}...", flush=True)
    try:
        AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        # Download config only (weights downloaded on first job run to save time)
        AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True,
            low_cpu_mem_usage=True)
        print(f"  OK: {mid}")
    except Exception as e:
        print(f"  FAIL {mid}: {e}")

print("\nDatasets...")
from datasets import load_dataset
for ds_id, cfg, split in [
    ("nyu-mll/blimp", "anaphor_gender_agreement", "train"),
    ("eriktks/conll2003", None, "train"),
    ("scan", "addprim_jump", "test"),
    ("metaeval/cogs", None, "gen"),
]:
    try:
        kw = {"trust_remote_code": True}
        if cfg: kw["name"] = cfg
        load_dataset(ds_id, split=split, **kw)
        print(f"  OK: {ds_id}")
    except Exception as e:
        print(f"  FAIL {ds_id}: {e}")
PYEOF

echo ""
echo "=== Download complete ==="
echo "Models cached at: $HF_HOME"
echo "Disk usage: $(du -sh $HF_HOME)"
