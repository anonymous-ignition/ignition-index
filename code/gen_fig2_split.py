"""
Generate two figures from existing PKL results:
1. fig2_representative.pdf — 3-panel figure for main body
   (Gemma2-2b, Huginn-3.5b, Mamba-2.8b on blimp_determiner_noun_agreement)
2. fig2_full_grid.pdf — full 12-model grid for appendix (same as existing fig2)

Run on Narval:
  cd $PROJECT/ignition_index/code
  python gen_fig2_split.py
"""

import pickle, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

RESULTS = Path(os.environ["SCRATCH"]) / "ignition_index" / "results"
OUTDIR  = Path(os.environ["SCRATCH"]) / "ignition_index" / "figures"
OUTDIR.mkdir(exist_ok=True)

SIGNAL_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TASK = "blimp_determiner_noun_agreement_1"

# Model display config: (pkl_key, display_name, arch_color)
ALL_MODELS = [
    ("gpt2_small",   "gpt2-small",   "FF",  "#1f77b4"),
    ("gpt2_medium",  "gpt2-medium",  "FF",  "#1f77b4"),
    ("gpt2_xl",      "gpt2-xl",      "FF",  "#1f77b4"),
    ("pythia_70m",   "pythia-70m",   "FF",  "#1f77b4"),
    ("pythia_410m",  "pythia-410m",  "FF",  "#1f77b4"),
    ("pythia_1p4b",  "pythia-1.4b",  "FF",  "#1f77b4"),
    ("pythia_6p9b",  "pythia-6.9b",  "FF",  "#1f77b4"),
    ("gemma2_2b",    "gemma2-2b",    "FF",  "#1f77b4"),
    ("gemma2_9b",    "gemma2-9b",    "FF",  "#1f77b4"),
    ("huginn_3p5b",  "huginn-3.5b",  "REC", "#d62728"),
    ("mamba_1p4b",   "mamba-1.4b",   "SSM", "#2ca02c"),
    ("mamba_2p8b",   "mamba-2.8b",   "SSM", "#2ca02c"),
]

REPRESENTATIVE = ["gemma2_2b", "huginn_3p5b", "mamba_2p8b"]
REP_LABELS = {
    "gemma2_2b":   r"Gemma 2 2B (FF, $\hat\beta=203.96$)",
    "huginn_3p5b": r"Huginn-3.5B (REC, $\hat\beta=111.03$)",
    "mamba_2p8b":  r"Mamba 2.8B (SSM, $\hat\beta=78.87$)",
}
REP_COLORS = {
    "gemma2_2b":   "#1f77b4",
    "huginn_3p5b": "#d62728",
    "mamba_2p8b":  "#2ca02c",
}

def sigmoid4(x, ymin, ymax, x0, beta):
    return ymin + (ymax - ymin) / (1 + np.exp(-beta * (x - x0)))

def load_layer_acc(pkl_key, task, slevel, signal_type="S1"):
    p = RESULTS / f"model_{pkl_key}.pkl"
    if not p.exists():
        p = RESULTS / f"model_{pkl_key}_partial.pkl"
    if not p.exists():
        return None, None
    r = pickle.load(open(p, "rb"))
    key = (task, signal_type, slevel)
    layer_accs = r.get("layer_accs", {}).get(key, None)
    if layer_accs is None:
        # fallback: reconstruct from stored per-layer arrays if available
        return None, None
    n_layers = r.get("n_layers", len(layer_accs))
    x = np.linspace(0, 1, len(layer_accs))
    return x, np.array(layer_accs)

def plot_model_panel(ax, pkl_key, color, title):
    """Plot all signal levels for one model on one task."""
    alphas = np.linspace(0.25, 1.0, len(SIGNAL_LEVELS))
    plotted = False
    for i, sl in enumerate(SIGNAL_LEVELS):
        x, acc = load_layer_acc(pkl_key, TASK, sl)
        if x is None or acc is None:
            continue
        ax.plot(x, acc * 100, color=color, alpha=alphas[i], linewidth=1.2)
        # Fit sigmoid and overlay
        try:
            p0 = [acc.min(), acc.max(),
                  x[np.searchsorted(acc, (acc.min()+acc.max())/2)], 5.0]
            popt, _ = curve_fit(sigmoid4, x, acc, p0=p0,
                                maxfev=5000, method="lm")
            xfit = np.linspace(0, 1, 200)
            ax.plot(xfit, sigmoid4(xfit, *popt)*100,
                    color=color, alpha=alphas[i], linewidth=0.7,
                    linestyle="--")
        except Exception:
            pass
        plotted = True

    ax.set_xlim(0, 1)
    ax.set_ylim(45, 105)
    ax.set_xlabel(r"Layer $\ell/L$", fontsize=7)
    ax.set_ylabel("Probe acc. (%)", fontsize=7)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    if not plotted:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")

# ── Figure 1: Representative 3-panel ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
for ax, key in zip(axes, REPRESENTATIVE):
    plot_model_panel(ax, key, REP_COLORS[key], REP_LABELS[key])

# Add signal strength colorbar legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color="gray", alpha=a, linewidth=1.5,
                  label=f"$s={sl:.1f}$")
           for a, sl in zip(np.linspace(0.25,1.0,6), SIGNAL_LEVELS)]
axes[-1].legend(handles=handles, fontsize=5.5, loc="lower right",
                framealpha=0.7, ncol=2)

fig.suptitle(r"Fig.~2: Per-Layer Probe Accuracy — Representative Models"
             "\n(task: blimp\\_determiner\\_noun\\_agreement\\_1)",
             fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.90])
out1 = OUTDIR / "fig2_representative.pdf"
fig.savefig(out1, dpi=300, bbox_inches="tight")
fig.savefig(str(out1).replace(".pdf",".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: Full 12-model grid ─────────────────────────────────
n_models = len(ALL_MODELS)
fig, axes = plt.subplots(n_models, 1, figsize=(14.0, n_models * 1.8))
if n_models == 1:
    axes = [axes]

for ax, (key, name, arch, color) in zip(axes, ALL_MODELS):
    plot_model_panel(ax, key, color, f"{name} ({arch})")

fig.suptitle("Fig.~2 (Full): Per-Layer Probe Accuracy — All 12 Models\n"
             "(task: blimp\\_determiner\\_noun\\_agreement\\_1)",
             fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = OUTDIR / "fig2_full_grid.pdf"
fig.savefig(out2, dpi=300, bbox_inches="tight")
fig.savefig(str(out2).replace(".pdf",".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")

print("Done.")
