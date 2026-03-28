"""
Generate NeurIPS-quality per-layer accuracy figures.
Run on Narval: python gen_fig2_v3.py
"""
import pickle, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────
RESULTS = Path(os.environ["SCRATCH"]) / "ignition_index" / "results"
OUTDIR  = Path(os.environ["SCRATCH"]) / "ignition_index" / "figures"
OUTDIR.mkdir(exist_ok=True)

SIGNAL_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TASK     = "blimp_determiner_noun_agreement_1"
SIG_TYPE = "S1"

# ── NeurIPS-style global settings ───────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   7.5,
    "xtick.labelsize":  6.5,
    "ytick.labelsize":  6.5,
    "axes.linewidth":   0.6,
    "grid.linewidth":   0.3,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.3,
    "legend.fontsize":  6.5,
    "legend.framealpha":0.9,
    "legend.edgecolor": "0.7",
    "figure.dpi":       150,
})

ARCH_COLORS = {
    "FF":  "#1f77b4",   # blue
    "REC": "#d62728",   # red
    "SSM": "#2ca02c",   # green
}

ALL_MODELS = [
    ("gpt2_small",  "GPT-2 Small\n(FF, 124M)",   "FF"),
    ("gpt2_medium", "GPT-2 Medium\n(FF, 355M)",  "FF"),
    ("gpt2_xl",     "GPT-2 XL\n(FF, 1.5B)",      "FF"),
    ("pythia_70m",  "Pythia-70M\n(FF)",           "FF"),
    ("pythia_410m", "Pythia-410M\n(FF)",          "FF"),
    ("pythia_1p4b", "Pythia-1.4B\n(FF)",          "FF"),
    ("pythia_6p9b", "Pythia-6.9B\n(FF)",          "FF"),
    ("gemma2_2b",   "Gemma 2 2B\n(FF, 2.6B)",    "FF"),
    ("gemma2_9b",   "Gemma 2 9B\n(FF, 9.2B)",    "FF"),
    ("huginn_3p5b", "Huginn-3.5B\n(REC)",         "REC"),
    ("mamba_1p4b",  "Mamba-1.4B\n(SSM)",          "SSM"),
    ("mamba_2p8b",  "Mamba-2.8B\n(SSM)",          "SSM"),
]

REPRESENTATIVE = [
    ("gemma2_2b",  "Gemma 2 2B\n" r"(Feedforward, $\hat\beta=203.96$)",  "FF"),
    ("huginn_3p5b","Huginn-3.5B\n" r"(Recurrent-depth, $\hat\beta=111.03$)", "REC"),
    ("mamba_2p8b", "Mamba-2.8B\n" r"(SSM, $\hat\beta=78.87$)",           "SSM"),
]

def sigmoid4(x, ymin, ymax, x0, beta):
    exponent = np.clip(-beta * (x - x0), -500, 500)
    return ymin + (ymax - ymin) / (1 + np.exp(exponent))

def load_pkl(pkl_key):
    for suffix in ["", "_partial"]:
        p = RESULTS / f"model_{pkl_key}{suffix}.pkl"
        if p.exists():
            return pickle.load(open(p, "rb"))
    return None

def plot_panel(ax, pkl_key, arch, title, show_ylabel=True, show_legend=False):
    color = ARCH_COLORS[arch]
    r = load_pkl(pkl_key)

    if r is None:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        ax.set_title(title, pad=5)
        return

    curves = r.get("acc_curves_disc", {})
    alphas = np.linspace(0.2, 1.0, len(SIGNAL_LEVELS))
    cmap_colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(SIGNAL_LEVELS))) \
                  if arch == "FF" else \
                  plt.cm.Reds(np.linspace(0.35, 0.95, len(SIGNAL_LEVELS))) \
                  if arch == "REC" else \
                  plt.cm.Greens(np.linspace(0.35, 0.95, len(SIGNAL_LEVELS)))

    plotted_lines = []
    for i, sl in enumerate(SIGNAL_LEVELS):
        key = (TASK, SIG_TYPE, sl)
        if key not in curves:
            continue
        acc = np.array(curves[key], dtype=float)
        x = np.linspace(0, 1, len(acc))
        line, = ax.plot(x, acc * 100, color=cmap_colors[i],
                        linewidth=1.4, zorder=3)
        plotted_lines.append((line, f"$s={sl:.1f}$"))

        # Sigmoid overlay (dashed, slightly thinner)
        try:
            mid_val = (acc.min() + acc.max()) / 2.0
            x0_init = float(x[np.argmin(np.abs(acc - mid_val))])
            p0 = [float(acc.min()), float(acc.max()), x0_init, 5.0]
            popt, _ = curve_fit(sigmoid4, x, acc, p0=p0,
                                maxfev=10000, method="lm")
            xfit = np.linspace(0, 1, 400)
            ax.plot(xfit, sigmoid4(xfit, *popt) * 100,
                    color=cmap_colors[i], linewidth=0.7,
                    linestyle="--", zorder=2, alpha=0.85)
        except Exception:
            pass

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(44, 103)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.set_xlabel(r"Normalised layer depth $\ell/L$")
    if show_ylabel:
        ax.set_ylabel("Probe accuracy (%)")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, pad=5, fontweight="bold")
    ax.grid(True, axis="both")
    ax.spines[["top", "right"]].set_visible(False)

    # Arch label in corner (small coloured tag)
    arch_label = {"FF":"Feedforward","REC":"Recurrent-depth","SSM":"SSM (no attn)"}[arch]
    ax.text(0.97, 0.04, arch_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.5,
            color=color, fontweight="bold")

    if show_legend and plotted_lines:
        handles = [l for l, _ in plotted_lines]
        labels  = [lb for _, lb in plotted_lines]
        ax.legend(handles, labels, loc="upper left",
                  title="Signal $s$", title_fontsize=6,
                  ncol=2, handlelength=1.2, columnspacing=0.8,
                  borderpad=0.5, labelspacing=0.3)

# ════════════════════════════════════════════════════════════════
# Figure A: 3-panel representative (main body)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6),
                         gridspec_kw={"wspace": 0.32})

for idx, (ax, (key, label, arch)) in enumerate(zip(axes, REPRESENTATIVE)):
    show_y = (idx == 0)
    show_leg = (idx == 2)   # legend on rightmost panel only
    plot_panel(ax, key, arch, label,
               show_ylabel=show_y, show_legend=show_leg)

# Single shared signal-strength legend BELOW all panels
from matplotlib.lines import Line2D
from matplotlib.cm import Blues, Reds, Greens
leg_handles = []
for i, sl in enumerate(SIGNAL_LEVELS):
    c = Blues(0.35 + 0.6 * i / 5)
    leg_handles.append(Line2D([0],[0], color=c, linewidth=1.5,
                               label=f"$s={sl:.1f}$"))
fig.legend(handles=leg_handles,
           title="Input signal strength $s$",
           title_fontsize=7,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.18),
           ncol=6,
           framealpha=0.9,
           edgecolor="0.7",
           fontsize=6.5)

fig.suptitle(
    "Per-Layer Probe Accuracy with Fitted Sigmoid Curves\n"
    r"(Task: \texttt{blimp\_determiner\_noun\_agreement\_1}, "
    "Signal type: S1 token masking)",
    fontsize=8, y=1.03)

for ext in ["pdf", "png"]:
    out = OUTDIR / f"fig2_representative.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ════════════════════════════════════════════════════════════════
# Figure B: Full 12-model grid (appendix)
# ════════════════════════════════════════════════════════════════
ncols = 4
nrows = 3   # 12 models exactly
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(13.0, nrows * 2.7),
                         gridspec_kw={"wspace": 0.30, "hspace": 0.52})
axes_flat = axes.flatten()

for i, (key, name, arch) in enumerate(ALL_MODELS):
    show_y   = (i % ncols == 0)
    show_leg = (i == 2)   # legend on third panel (top-right area)
    plot_panel(axes_flat[i], key, arch, name,
               show_ylabel=show_y, show_legend=show_leg)

# Hide unused panels
for j in range(len(ALL_MODELS), len(axes_flat)):
    axes_flat[j].set_visible(False)

# Shared legend at bottom
fig.legend(handles=leg_handles,
           title="Input signal strength $s$",
           title_fontsize=7.5,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.03),
           ncol=6,
           framealpha=0.9,
           edgecolor="0.7",
           fontsize=7)

fig.suptitle(
    "Per-Layer Probe Accuracy with Fitted Sigmoid Curves — All 12 Models\n"
    r"(Task: \texttt{blimp\_determiner\_noun\_agreement\_1}, "
    "Signal type: S1 token masking)",
    fontsize=9, y=1.02)

for ext in ["pdf", "png"]:
    out = OUTDIR / f"fig2_full_grid.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

print("All done.")
