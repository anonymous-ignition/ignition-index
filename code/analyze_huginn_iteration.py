"""
analyze_huginn_iteration.py — Compare depth-axis vs iteration-axis β̂ for Huginn

Loads:
  - model_huginn-v4.5-3.5b.pkl (depth-axis probing)
  - huginn_iteration_probing.pkl (iteration-axis probing)

Outputs:
  - Comparison statistics
  - Figure for paper (depth vs iteration β̂)
  - LaTeX table
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

RESULTS_DIR = Path(os.environ['SCRATCH']) / 'ignition_index' / 'results'
FIGURES_DIR = Path(os.environ['SCRATCH']) / 'ignition_index' / 'figures'

# ── Load results ───────────────────────────────────────────────────
def load_results():
    """Load both depth and iteration probing results."""
    depth_path = RESULTS_DIR / 'model_huginn-v4.5-3.5b.pkl'
    iter_path = RESULTS_DIR / 'huginn_iteration_probing.pkl'
    
    with open(depth_path, 'rb') as f:
        depth_data = pickle.load(f)
    
    with open(iter_path, 'rb') as f:
        iter_data = pickle.load(f)
    
    return depth_data, iter_data

# ── Comparison statistics ──────────────────────────────────────────
def compute_statistics(depth_data, iter_data):
    """Compute β̂ statistics for depth vs iteration."""
    
    # Extract all beta values
    depth_betas = []
    iter_betas = []
    
    for key in depth_data['beta_hat']:
        beta_d = depth_data['beta_hat'][key]
        if key in iter_data['beta_hat']:
            beta_i = iter_data['beta_hat'][key]
            
            # Exclude ceiling hits
            if beta_d < 300 and beta_i < 300:
                if not np.isnan(beta_d) and not np.isnan(beta_i):
                    depth_betas.append(beta_d)
                    iter_betas.append(beta_i)
    
    depth_mean = np.mean(depth_betas)
    iter_mean = np.mean(iter_betas)
    
    print("="*70)
    print("HUGINN: DEPTH-AXIS vs ITERATION-AXIS COMPARISON")
    print("="*70)
    print(f"Depth-axis β̂:     {depth_mean:.1f} ± {np.std(depth_betas):.1f}")
    print(f"Iteration-axis β̂: {iter_mean:.1f} ± {np.std(iter_betas):.1f}")
    print(f"Ratio (iter/depth): {iter_mean/depth_mean:.2f}×")
    print(f"Difference: {iter_mean - depth_mean:+.1f}")
    print(f"N comparisons: {len(depth_betas)}")
    print("="*70)
    
    return {
        'depth_betas': depth_betas,
        'iter_betas': iter_betas,
        'depth_mean': depth_mean,
        'iter_mean': iter_mean,
    }

# ── Generate figure ────────────────────────────────────────────────
def generate_figure(stats):
    """Generate comparison figure for paper."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: scatter plot depth vs iteration
    ax1.scatter(stats['depth_betas'], stats['iter_betas'], alpha=0.6, s=50)
    ax1.plot([0, 300], [0, 300], 'k--', alpha=0.3, label='y=x')
    ax1.set_xlabel('Depth-axis β̂', fontsize=12)
    ax1.set_ylabel('Iteration-axis β̂', fontsize=12)
    ax1.set_title('Huginn: Depth vs Iteration β̂', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right panel: bar comparison
    means = [stats['depth_mean'], stats['iter_mean']]
    stds = [np.std(stats['depth_betas']), np.std(stats['iter_betas'])]
    labels = ['Depth-axis\n(layers)', 'Iteration-axis\n(recurrent passes)']
    
    bars = ax2.bar(labels, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e'])
    ax2.set_ylabel('Mean β̂', fontsize=12)
    ax2.set_title('Huginn Ignition Index', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'huginn_depth_vs_iteration.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    
    return output_path

# ── Generate LaTeX table ───────────────────────────────────────────
def generate_latex_table(stats):
    """Generate LaTeX table for paper integration."""
    
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)
    
    print("""
\\begin{table}[h]
\\centering
\\caption{\\textbf{Huginn architectural decomposition: depth vs iteration axis.}}
\\label{tab:huginn_iteration}
\\small
\\begin{tabular}{lrrr}
\\toprule
\\textbf{Probe axis} & $\\overline{\\hat{\\beta}}$ & \\textbf{95\\% CI} & $\\Delta w$ \\\\
\\midrule
""")
    
    print(f"Depth (layers 0--31)           & {stats['depth_mean']:6.1f} & [--] & 85.0 \\\\")
    print(f"Iteration (recurrent passes 1--64) & {stats['iter_mean']:6.1f} & [--] & -- \\\\")
    
    print("""\\midrule
\\multicolumn{4}{l}{\\textit{Ratio (iteration / depth)}} \\\\
""")
    
    ratio = stats['iter_mean'] / stats['depth_mean']
    print(f"\\multicolumn{{4}}{{l}}{{\\quad {ratio:.2f}$\\times$}} \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nLoading results...")
    depth_data, iter_data = load_results()
    
    print("\nComputing statistics...")
    stats = compute_statistics(depth_data, iter_data)
    
    print("\nGenerating figure...")
    generate_figure(stats)
    
    print("\nGenerating LaTeX table...")
    generate_latex_table(stats)
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
If iteration-axis β̂ >> depth-axis β̂:
  → H1 REINTERPRETATION: Huginn's ignition operates along iteration axis
  → Paper update: "Huginn's low depth β̂ reflects architectural choice—
     ignition concentrates across recurrent iterations, not depth"
  → DISCOVERY: First demonstration of axis-dependent ignition structure

If iteration-axis β̂ ≈ depth-axis β̂:
  → Huginn genuinely has lower ignition than FF models
  → Confirms current interpretation (recurrence alone insufficient)

If iteration-axis β̂ < depth-axis β̂:
  → Unexpected—would require deeper investigation
""")
    
    print("\nAnalysis complete.")
