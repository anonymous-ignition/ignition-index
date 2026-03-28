"""
gen_appendix_figures.py — Generate NeurIPS 2026 Appendix B figures.

Creates three full-grid figures (one per task) showing all 12 models:
  - Figure B.1: Determiner-Noun Agreement
  - Figure B.2: NPI Licensing  
  - Figure B.3: Principle A C-command

Each figure is sized for full-page appendix layout with readable fonts.

Usage:
    python gen_appendix_figures.py --signal_type S1
    python gen_appendix_figures.py --signal_type S1 --tasks det_noun npi c_command
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    MODEL_REGISTRY, RESULTS_DIR, FIGURES_DIR, LOGS_DIR
)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "gen_appendix_figures.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Architecture styling (matches aggregate.py)
ARCH_COLOR = {"FF": "#1f77b4", "REC": "#d62728", "SSM": "#2ca02c"}
ARCH_LABEL = {"FF": "Feedforward", "REC": "Recurrent-depth", "SSM": "SSM (no attn)"}

# Signal levels to plot
SIGNAL_LEVELS = [0.2, 0.6, 1.0]

# Task configuration
TASK_CONFIG = {
    "det_noun": {
        "display": "Determiner-Noun Agreement",
        "key": "blimp_determiner_noun_agreement_1",
        "label": "B.1"
    },
    "npi": {
        "display": "NPI Licensing",
        "key": "blimp_npi_present_1",
        "label": "B.2"
    },
    "c_command": {
        "display": "Principle A C-command",
        "key": "blimp_principle_A_c_command",
        "label": "B.3"
    },
}

# ── IO helpers (matches aggregate.py pattern) ────────────────────
def load_pkl(name):
    """Load pickle file from RESULTS_DIR."""
    p = RESULTS_DIR / f"{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def load_model_data(model_key):
    """Load model results with standardized naming."""
    safe = f"model_{model_key.replace('-', '_').replace('.', 'p')}"
    return load_pkl(safe)


# ── Plotting functions ───────────────────────────────────────────
def get_task_beta(data, task_key, signal_type, signal_level):
    """Extract beta_hat for specific task/signal combination."""
    key = (task_key, signal_type, signal_level)
    beta_dict = data.get('beta_hat', {})
    
    if key in beta_dict:
        beta = float(beta_dict[key])
        # Cap at 300 (some fits hit numerical limits)
        return min(beta, 300.0)
    return None


def get_task_curves(data, task_key, signal_type):
    """Extract accuracy curves for all signal levels."""
    acc_curves = data.get('acc_curves_disc', {})
    
    curves = {}
    for s in SIGNAL_LEVELS:
        key = (task_key, signal_type, s)
        if key in acc_curves:
            curves[s] = acc_curves[key]
    
    return curves if curves else None


def plot_model_panel(ax, model_key, task_key, signal_type):
    """Plot single model panel with probe curves."""
    
    # Load data
    data = load_model_data(model_key)
    
    # Get display name and architecture
    if model_key not in MODEL_REGISTRY:
        ax.text(0.5, 0.5, 'Unknown model', ha='center', va='center',
                fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(45, 100)
        return
    
    model_info = MODEL_REGISTRY[model_key]
    display_name = model_key.replace('-', ' ').replace('_', ' ').title()
    arch = model_info[3]  # Architecture code (FF/REC/SSM)
    
    if data is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(45, 100)
        ax.set_xlabel('Layer $\ell/L$', fontsize=9)
        ax.set_ylabel('Probe acc. (%)', fontsize=9)
        ax.set_title(f"{display_name}\n(missing)", fontsize=10)
        return
    
    # Get curves for this task
    curves = get_task_curves(data, task_key, signal_type)
    
    if curves is None or len(curves) == 0:
        ax.text(0.5, 0.5, 'No task data', ha='center', va='center',
                fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(45, 100)
        ax.set_xlabel('Layer $\ell/L$', fontsize=9)
        ax.set_ylabel('Probe acc. (%)', fontsize=9)
        ax.set_title(f"{display_name}\n(no data)", fontsize=10)
        return
    
    # Get representative beta (use s=1.0)
    beta_hat = get_task_beta(data, task_key, signal_type, 1.0)
    if beta_hat is None:
        beta_hat = 0.0
    
    # Number of layers
    n_layers = len(list(curves.values())[0])
    x = np.linspace(0, 1, n_layers)
    color = ARCH_COLOR.get(arch, "#888")
    
    # Plot curves for each signal level (matches aggregate.py style)
    for sl, alpha_ in [(1.0, 1.0), (0.6, 0.55), (0.2, 0.25)]:
        if sl not in curves:
            continue
        
        # Convert to percentages
        y = np.array(curves[sl]) * 100
        
        ax.plot(x, y, color=color, alpha=alpha_, 
                linewidth=1.8 + sl * 0.7)  # Thicker for print
    
    # Logistic fit curve (approximation)
    if beta_hat > 0:
        y_fit = 50 + 45 / (1 + np.exp(-beta_hat * (x - 0.3)))
        ax.plot(x, y_fit, '--', color=color, linewidth=2.2, alpha=0.7)
    
    # Styling (NeurIPS-sized fonts)
    ax.set_xlim(0, 1)
    ax.set_ylim(45, 100)
    ax.set_xlabel('Layer $\ell/L$', fontsize=9)
    ax.set_ylabel('Probe acc. (%)', fontsize=9)
    
    # Title with β̂
    title = f"{display_name}\n$\\hat{{\\beta}}={beta_hat:.1f}$"
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8)


def generate_task_figure(task_short, task_info, signal_type):
    """Generate one appendix figure for a single task across all models."""
    
    task_display = task_info["display"]
    task_key = task_info["key"]
    fig_label = task_info["label"]
    
    log.info(f"Generating Figure {fig_label}: {task_display}")
    
    # Get available models (matches aggregate.py pattern)
    avail_models = [mk for mk in MODEL_REGISTRY if load_model_data(mk) is not None]
    
    if not avail_models:
        log.info("  WARNING: No models with data found")
        return None
    
    log.info(f"  Found {len(avail_models)} models with data")
    
    # 4x3 grid (12 models)
    fig, axes = plt.subplots(4, 3, figsize=(11, 13))
    axes = axes.flatten()
    
    # Plot each model
    for idx, model_key in enumerate(sorted(avail_models)[:12]):
        if idx < len(axes):
            plot_model_panel(axes[idx], model_key, task_key, signal_type)
    
    # Hide unused subplots
    for idx in range(len(avail_models), len(axes)):
        axes[idx].axis('off')
    
    # Main title
    fig.suptitle(f"Figure {fig_label}: All Models on {task_display}",
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Legend at bottom
    s_handles = [
        mpatches.Patch(color='gray', alpha=0.25, label='s=0.2'),
        mpatches.Patch(color='gray', alpha=0.55, label='s=0.6'),
        mpatches.Patch(color='gray', alpha=1.0, label='s=1.0'),
        mpatches.Patch(color='gray', alpha=0.7, label='Fit'),
    ]
    
    arch_handles = [
        mpatches.Patch(color=ARCH_COLOR["FF"], label=ARCH_LABEL["FF"]),
        mpatches.Patch(color=ARCH_COLOR["REC"], label=ARCH_LABEL["REC"]),
        mpatches.Patch(color=ARCH_COLOR["SSM"], label=ARCH_LABEL["SSM"]),
    ]
    
    fig.legend(
        handles=s_handles + arch_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.015),
        ncol=7,
        frameon=True,
        fontsize=11,
        columnspacing=1.2,
    )
    
    plt.tight_layout(rect=[0, 0.025, 1, 0.99])
    
    return fig, task_short, fig_label


def save_figure(fig, name):
    """Save figure in PDF and PNG (matches aggregate.py pattern)."""
    # Create appendix subdirectory
    appendix_dir = FIGURES_DIR / "appendix"
    appendix_dir.mkdir(exist_ok=True, parents=True)
    
    for fmt in ["pdf", "png"]:
        output_path = appendix_dir / f"{name}.{fmt}"
        fig.savefig(output_path, bbox_inches="tight",
                    dpi=600 if fmt == "png" else None)
    
    plt.close(fig)
    log.info(f"  Saved {name}.pdf/.png")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS 2026 Appendix B figures"
    )
    parser.add_argument(
        "--signal_type", default="S1",
        help="Signal type to use (default: S1)"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["det_noun", "npi", "c_command"],
        choices=list(TASK_CONFIG.keys()),
        help="Tasks to generate figures for"
    )
    args = parser.parse_args()
    
    # Set matplotlib style (matches aggregate.py)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "pdf.fonttype": 42
    })
    
    log.info("="*70)
    log.info("Generating NeurIPS Appendix B Figures")
    log.info("="*70)
    log.info(f"Signal type: {args.signal_type}")
    log.info(f"Tasks: {', '.join(args.tasks)}")
    log.info(f"Results from: {RESULTS_DIR}")
    log.info(f"Figures to: {FIGURES_DIR / 'appendix'}")
    log.info("")
    
    # Generate figures for each task
    for task_short in args.tasks:
        task_info = TASK_CONFIG[task_short]
        result = generate_task_figure(task_short, task_info, args.signal_type)
        
        if result is not None:
            fig, task_name, fig_label = result
            filename = f"fig{fig_label.replace('.', '')}__{task_info['key']}"
            save_figure(fig, filename)
        else:
            log.info(f"  Skipped {task_short} (no data)")
        log.info("")
    
    log.info("="*70)
    log.info("✓ Appendix figure generation complete")
    log.info("="*70)
