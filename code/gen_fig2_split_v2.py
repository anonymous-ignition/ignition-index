"""
Generate two figures from existing PKL results:
1. fig2_representative.pdf — 3-panel for main body
2. fig2_full_grid.pdf — full 12-model grid for appendix

Run on Narval:
  cd $PROJECT/ignition_index/code
  python gen_fig2_split_v2.py
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
SIG_TYPE = "S1"

ALL_MODELS = [
    ("gpt2_small",   "GPT-2 Small",   "#1f77b4"),
    ("gpt2_medium",  "GPT-2 Medium",  "#1f77b4"),
    ("gpt2_xl",      "GPT-2 XL",      "#1f77b4"),
    ("pythia_70m",   "Pythia-70M",    "#1f77b4"),
    ("pythia_410m",  "Pythia-410M",   "#1f77b4"),
    ("pythia_1p4b",  "Pythia-1.4B",   "#1f77b4"),
    ("pythia_6p9b",  "Pythia-6.9B",   "#1f77b4"),
    ("gemma2_2b",    "Gemma 2 2B",    "#1f77b4"),
    ("gemma2_9b",    "Gemma 2 9B",    "#1f77b4"),
    ("huginn_3p5b",  "Huginn-3.5B",   "#d62728"),
    ("mamba_1p4b",   "Mamba-1.4B",    "#2ca02c"),
    ("mamba_2p8b",   "Mamba-2.8B",    "#2ca02c"),
]

REPRESENTATIVE = [
    ("gemma2_2b",   r"Gemma 2 2B  (FF, $\hat\beta=203.96$)",  "#1f77b4"),
    ("huginn_3p5b", r"Huginn-3.5B  (REC, $\hat\beta=111.03$)", "#d62728"),
    ("mamba_2p8b",  r"Mamba-2.8B  (SSM, $\hat\beta=78.87$)",   "#2ca02c"),
]

def sigmoid4(x, ymin, ymax, x0, beta):
    return ymin + (ymax - ymin) / (1 + np.exp(-beta * (x - x0)))

def load_curves(pkl_key):
    for suffix in ["", "_partial"]:
        p = RESULTS / f"model_{pkl_key}{suffix}.pkl"
        if p.exists():
            return pickle.load(open(p, "rb"))
    return None

def plot_panel(ax, pkl_key, color, title):
    r = load_curves(pkl_key)
    if r is None:
        ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        ax.set_title(title, fontsize=8)
        return

    curves = r.get("acc_curves_disc", {})
    n_layers = r.get("n_layers", None)
    alphas = np.linspace(0.25, 1.0, len(SIGNAL_LEVELS))

    for i, sl in enumerate(SIGNAL_LEVELS):
        key = (TASK, SIG_TYPE, sl)
        if key not in curves:
            continue
        acc = np.array(curves[key], dtype=float)
        if n_layers is None:
            n_layers = len(acc)
        x = np.linspace(0, 1, len(acc))
        ax.plot(x, acc * 100, color=color, alpha=alphas[i], linewidth=1.4)
        # Sigmoid overlay
        try:
            mid_val = (acc.min() + acc.max()) / 2
            x0_init = float(x[np.argmin(np.abs(acc - mid_val))])
            p0 = [float(acc.min()), float(acc.max()), x0_init, 5.0]
            popt, _ = curve_fit(sigmoid4, x, acc, p0=p0, maxfev=8000, method="lm")
            xfit = np.linspace(0, 1, 300)
            ax.plot(xfit, sigmoid4(xfit, *popt) * 100,
                    color=color, alpha=alphas[i] * 0.7,
                    linewidth=0.8, linestyle="--")
        except Exception:
            pass

    ax.set_xlim(0, 1)
    ax.set_ylim(44, 106)
    ax.set_xlabel(r"Layer $\ell/L$", fontsize=7)
    ax.set_ylabel("Probe acc. (%)", fontsize=7)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines[["top","right"]].set_visible(False)

# ── Figure A: 3-panel representative ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, (key, label, color) in zip(axes, REPRESENTATIVE):
    plot_panel(ax, key, color, label)

# Signal strength legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color="gray", alpha=a, linewidth=1.5,
                  label=f"$s={sl:.1f}$")
           for a, sl in zip(np.linspace(0.25,1.0,6), SIGNAL_LEVELS)]
axes[2].legend(handles=handles, fontsize=5.5, loc="lower right",
               framealpha=0.8, ncol=2, borderpad=0.4)

fig.suptitle("Per-Layer Probe Accuracy (task: determiner–noun agreement)",
             fontsize=8.5, y=1.01)
plt.tight_layout()
for ext in ["pdf", "png"]:
    out = OUTDIR / f"fig2_representative.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ── Figure B: Full 12-model grid ─────────────────────────────────
n = len(ALL_MODELS)
ncols = 4
nrows = (n + ncols - 1) // ncols   # 3 rows of 4
fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, nrows * 2.8))
axes_flat = axes.flatten()

for i, (key, name, color) in enumerate(ALL_MODELS):
    plot_panel(axes_flat[i], key, color, name)

# Hide unused panels
for j in range(len(ALL_MODELS), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("Per-Layer Probe Accuracy — All 12 Models\n"
             "(task: determiner–noun agreement, signal type S1)",
             fontsize=10, y=1.01)
plt.tight_layout()
for ext in ["pdf", "png"]:
    out = OUTDIR / f"fig2_full_grid.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

print("Done.")
