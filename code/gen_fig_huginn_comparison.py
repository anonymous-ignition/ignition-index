import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set publication quality
rcParams['font.size'] = 10
rcParams['font.family'] = 'serif'
rcParams['figure.dpi'] = 300

# Load data from scratch
import os
SCRATCH = os.environ['SCRATCH']

with open(f'{SCRATCH}/ignition_index/results/model_huginn_3p5b.pkl', 'rb') as f:
    depth_data = pickle.load(f)

with open(f'{SCRATCH}/ignition_index/results/huginn_iteration_probing.pkl', 'rb') as f:
    iter_data = pickle.load(f)

# Extract s=1.0 determiner-noun task for both
task_key = ('blimp_determiner_noun_agreement_1', 'S1', 1.0)

# Depth-axis
depth_acc = depth_data['acc_curves_disc'][task_key]
depth_beta = depth_data['beta_hat'][task_key]
depth_layers = np.arange(len(depth_acc)) / (len(depth_acc) - 1)

# Iteration-axis  
iter_acc = iter_data['acc_curves_disc'][task_key]
iter_beta = iter_data['beta_hat'][task_key]
iter_iters = np.arange(len(iter_acc)) / (len(iter_acc) - 1)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Depth-axis panel
ax1.plot(depth_layers, depth_acc, 'o-', color='#d62728', alpha=0.7, markersize=3)
ax1.set_xlabel('Normalized layer depth')
ax1.set_ylabel('Probe accuracy')
ax1.set_title(f'Depth-axis (β̂={depth_beta:.1f})', fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_ylim([0.45, 1.05])

# Iteration-axis panel
ax2.plot(iter_iters, iter_acc, 'o-', color='#d62728', alpha=0.7, markersize=3)
ax2.set_xlabel('Normalized iteration')
ax2.set_ylabel('Probe accuracy')
ax2.set_title(f'Iteration-axis (β̂={iter_beta:.1f})', fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_ylim([0.45, 1.05])

plt.suptitle('Huginn-3.5B: Iteration-axis vs Depth-axis Ignition\n(Determiner-Noun Agreement, s=1.0)', 
             fontsize=12, fontweight='bold')
plt.tight_layout()

# Create output directory if needed
os.makedirs('figures', exist_ok=True)

# Save
plt.savefig('figures/fig_huginn_comparison.pdf', bbox_inches='tight')
plt.savefig('figures/fig_huginn_comparison.png', bbox_inches='tight', dpi=300)
print("Saved: figures/fig_huginn_comparison.pdf")
print(f"Depth β̂={depth_beta:.1f}, Iteration β̂={iter_beta:.1f}, Ratio={iter_beta/depth_beta:.2f}×")
