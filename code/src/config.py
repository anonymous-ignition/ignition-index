"""
config.py — All hyperparameters in one place.
Paths hardcoded for AUTHOR on Narval (Compute Canada).
  PROJECT = /home/AUTHOR/projects/ACCOUNT/AUTHOR
  SCRATCH = /home/AUTHOR/scratch
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
SCRATCH     = Path(os.environ.get("SCRATCH",  "/home/AUTHOR/scratch"))
PROJECT     = Path(os.environ.get("PROJECT",  "/home/AUTHOR/projects/ACCOUNT/AUTHOR"))

RESULTS_DIR = SCRATCH / "ignition_index" / "results"
FIGURES_DIR = SCRATCH / "ignition_index" / "figures"
TABLES_DIR  = SCRATCH / "ignition_index" / "tables"
LOGS_DIR    = SCRATCH / "ignition_index" / "logs"
HF_CACHE    = PROJECT / "ignition_index" / "hf_cache"

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, HF_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sec 4.2: Signal levels ────────────────────────────────────────
SIGNAL_LEVELS  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONTENT_POS    = {"NOUN", "VERB", "ADJ", "ADV"}

# ── Sec 4.3: Probing ─────────────────────────────────────────────
N_PROBE_SAMPLES = 1000
PROBE_CV_FOLDS  = 5
PROBE_MAX_ITER  = 500
PROBE_C_GRID    = [0.01, 0.1, 1.0, 10.0]

# ── Sec 4.4: Bootstrap ───────────────────────────────────────────
N_BOOTSTRAP  = 2000
ALPHA        = 0.05

# ── Schaeffer DeltaAICc threshold ────────────────────────────────
AICC_THRESHOLD = 4.0

# ── H5: Permutation p ────────────────────────────────────────────
N_PERM = 10_000

# ── H4: PELT ─────────────────────────────────────────────────────
PELT_PENALTY = 3.0

# ── BLiMP paradigms ──────────────────────────────────────────────
BLIMP_PARADIGMS = [
    "regular_plural_subject_verb_agreement_1",
    "determiner_noun_agreement_1",
    "principle_A_c_command",
    "wh_island",
    "npi_present_1",
]

# ── Model registry ────────────────────────────────────────────────
# (hf_id, n_layers, d_model, arch_class, use_tl)
MODEL_REGISTRY = {
    "gpt2-small":  ("gpt2",                              12, 768,  "FF",  True),
    "gpt2-medium": ("gpt2-medium",                       24, 1024, "FF",  True),
    "gpt2-xl":     ("gpt2-xl",                           48, 1600, "FF",  True),
    "pythia-70m":  ("EleutherAI/pythia-70m",              6,  512, "FF",  True),
    "pythia-410m": ("EleutherAI/pythia-410m",            24, 1024, "FF",  True),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b",            24, 2048, "FF",  True),
    "pythia-6.9b": ("EleutherAI/pythia-6.9b",            32, 4096, "FF",  True),
    "gemma2-2b":   ("google/gemma-2-2b",                 26, 2304, "FF",  True),
    "gemma2-9b":   ("google/gemma-2-9b",                 42, 3584, "FF",  True),
    "huginn-3.5b": ("tomg-group-umd/huginn-0125",        64, 2560, "REC", False),
    "mamba-1.4b":  ("state-spaces/mamba-1.4b-hf",       48, 2048, "SSM", False),
    "mamba-2.8b":  ("state-spaces/mamba-2.8b-hf",       64, 2560, "SSM", False),
}
PARAM_COUNTS = {
    "gpt2-small": 124e6,  "gpt2-medium": 355e6,   "gpt2-xl": 1500e6,
    "pythia-70m":  70e6,  "pythia-410m": 410e6,   "pythia-1.4b": 1400e6,
    "pythia-6.9b": 6900e6,
    "gemma2-2b": 2600e6,  "gemma2-9b": 9200e6,
    "huginn-3.5b": 3500e6,
    "mamba-1.4b": 1400e6, "mamba-2.8b": 2800e6,
}

# ── Pythia training checkpoints (H4) ─────────────────────────────
PYTHIA_CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000,
]
