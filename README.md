# The Ignition Index: Measuring Global Workspace Dynamics in Language Models

**Anonymous submission to TMLR**

This repository contains code and results for measuring Global Workspace Theory-aligned representational transitions in transformer language models.

## Paper Abstract

We introduce the Ignition Index (β̂), a validated scalar metric that operationalizes Global Workspace Theory's all-or-none ignition prediction in transformer language models. Across 12 models spanning five architecture families, shuffled-label controls validate 9.6-fold selectivity (p<0.001) for genuine linguistic structure. Key findings: (1) feedforward transformers show 89% higher ignition than state-space models (p<10⁻¹³); (2) Mamba exhibits near-linear layer profiles consistent with absent global broadcast; (3) Pythia-410M exhibits training-phase transition at step 256, earlier than induction-head formation.

## Repository Structure
```
.
├── code/
│   ├── src/               # Core modules
│   │   ├── config.py      # Model registry, task configs
│   │   ├── datasets.py    # BLiMP, CoNLL, UD loaders
│   │   ├── probing.py     # Sigmoid fitting, statistics
│   │   └── signals.py     # Signal manipulation
│   ├── run_model.py       # Main probing script
│   ├── aggregate.py       # Combine results across models
│   ├── gen_figures_final.py  # Generate all paper figures
│   └── requirements.txt
├── jobs/                  # SLURM job scripts
├── results/               # Pre-computed results (PKL files)
└── figures/               # Generated figures
```

## Installation

**Requirements:** Python 3.11, CUDA 12.2
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r code/requirements.txt
```

**Key dependencies:** PyTorch 2.0+, TransformerLens, HuggingFace Transformers, scikit-learn, scipy, ruptures

## Data

Download benchmark datasets:
```bash
cd code
bash download_data.sh
```

This downloads:
- BLiMP paradigms (5 tasks)
- CoNLL-2003 NER
- Universal Dependencies EN-EWT v2.13

## Reproducing Results

### 1. Run probing for a single model
```bash
python code/run_model.py \
    --model_key gpt2-small \
    --signal_types S1 \
    --output_dir results/
```

**Available models:** `gpt2-small`, `gpt2-medium`, `gpt2-xl`, `pythia-70m`, `pythia-410m`, `pythia-1.4b`, `pythia-6.9b`, `gemma2-2b`, `gemma2-9b`, `huginn-3.5b`, `mamba-1.4b`, `mamba-2.8b`

### 2. Run shuffled-label controls (T1.1)
```bash
python code/run_model.py \
    --model_key gpt2-small \
    --signal_types S1 \
    --shuffle_labels \
    --output_dir results/
```

### 3. Aggregate results
```bash
python code/aggregate.py --results_dir results/
```

Generates `GLOBAL_RESULTS.pkl` with all β̂ estimates, CIs, and statistics.

### 4. Generate figures
```bash
python code/gen_figures_final.py --results_pkl results/GLOBAL_RESULTS.pkl
```

Outputs:
- `fig2_shuffled_validation.pdf` (Figure 2)
- `fig3_representative_curves.pdf` (Figure 3)
- `fig4_ignition_by_model.pdf` (Figure 4)
- `fig5_signal_strength.pdf` (Figure 5)
- `fig6_training_dynamics.pdf` (Figure 6)
- Appendix figures (Figures 7-9)

## Hardware Requirements

- **GPU:** NVIDIA A100 40GB recommended (or V100 32GB)
- **Memory:** 48GB RAM minimum
- **Storage:** ~50GB for cached models + datasets
- **Compute:** ~380 GPU-hours for full 12-model suite

**Per-model estimates:**
- GPT-2 Small/Medium: 9-14h
- Pythia-6.9B: 11h
- Gemma2-9B: 45h
- Huginn-3.5B: 48h

## Pre-computed Results

All results (β̂ estimates, probe accuracies, fitted sigmoids) are included in `results/*.pkl`. To regenerate figures without re-running experiments:
```bash
python code/gen_figures_final.py --results_pkl results/GLOBAL_RESULTS.pkl
```

## Citation
```bibtex
@article{anonymous2026ignition,
  title={The Ignition Index: Measuring Global Workspace Dynamics in Language Models},
  author={Anonymous},
  journal={Under review at TMLR},
  year={2026}
}
```

## License

Code released under MIT License. Results and figures licensed under CC-BY 4.0.
