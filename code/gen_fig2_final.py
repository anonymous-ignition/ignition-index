"""
gen_fig2_final.py
Generates two clean NeurIPS-quality figures using exact same plotting
logic as aggregate.py, but with:
  - NO suptitle (LaTeX caption is the only caption)
  - Legend placed OUTSIDE panels (below figure)
  - Representative 3-panel for main body
  - Full grid for appendix (models as rows, tasks as columns)

Run on Narval:
  cd $PROJECT/ignition_index/code
  python gen_fig2_final.py
"""
import pickle, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config import MODEL_REGISTRY, SIGNAL_LEVELS
from src.probing import fit_sigmoid, sigmoid_4p

RESULTS = Path(os.environ["SCRATCH"]) / "ignition_index" / "results"
OUTDIR  = Path(os.environ["SCRATCH"]) / "ignition_index" / "figures"
OUTDIR.mkdir(exist_ok=True)

SIG_TYPE = "S1"
ARCH_COLOR = {"FF": "#1f77b4", "REC": "#d62728", "SSM": "#2ca02c"}
ARCH_LABEL = {"FF": "Feedforward", "REC": "Recurrent-depth", "SSM": "SSM (no attn)"}

# NeurIPS style
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,  # embeds fonts properly for submission
    "ps.fonttype":   42,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.3,
    "grid.alpha":     0.25,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

def load_pkl(name):
    for suffix in ["", "_partial"]:
        p = RESULTS / f"model_{name}{suffix}.pkl"
        if p.exists():
            return pickle.load(open(p, "rb"))
    return None

def get_avail_and_tasks():
    avail = []
    for mk in MODEL_REGISTRY:
        safe = f"model_{mk.replace('-','_').replace('.','p')}"
        r = load_pkl(safe)
        if r is not None:
            avail.append(mk)
    # get first available model's tasks
    safe0 = f"model_{avail[0].replace('-','_').replace('.','p')}"
    r0 = load_pkl(safe0)
    probe_keys = [k for k in r0["acc_curves_disc"] if k[1] == SIG_TYPE and k[2] == 1.0]
    task_keys = sorted(set(k[0] for k in probe_keys))[:3]
    return avail, task_keys

def plot_model_task(ax, mk, tk, show_xlabel=True, show_ylabel=True,
                    show_legend=False, small=False):
    """Exact same logic as aggregate.py build_figures, cleaned up."""
    res  = load_model(mk)
    if res is None:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=8)
        return

    col  = ARCH_COLOR.get(MODEL_REGISTRY[mk][3], "#888")
    arch = MODEL_REGISTRY[mk][3]
    nl   = res.get("n_layers", 12)
    ln   = np.linspace(0, 1, nl)

    # 3 representative signal levels with varying alpha — same as aggregate.py
    plot_levels = [(1.0, 1.0), (0.6, 0.55), (0.2, 0.25)]

    for sl, alpha_ in plot_levels:
        k_ = (tk, SIG_TYPE, sl)
        if k_ not in res["acc_curves_disc"]:
            continue
        accs_ = np.array(res["acc_curves_disc"][k_]) * 100
        lw = 1.5 if sl == 1.0 else 1.0
        ax.plot(ln, accs_, color=col, alpha=alpha_, linewidth=lw)
        if sl == 1.0:
            ft = fit_sigmoid(ln, accs_ / 100, return_all=True)
            if ft.get("popt") is not None:
                yf = sigmoid_4p(ln, *ft["popt"]) * 100
                ax.plot(ln, yf, "--", color=col, alpha=0.75, linewidth=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(44, 103)
    ax.set_yticks([50, 70, 90])
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(True)
    ax.spines[["top","right"]].set_visible(False)

    fs = 7 if small else 8
    if show_xlabel:
        ax.set_xlabel(r"Layer $\ell/L$", fontsize=fs)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Probe acc. (%)", fontsize=fs)
    else:
        ax.set_yticklabels([])

    # β annotation top-left
    bv = res["beta_hat"].get((tk, SIG_TYPE, 1.0), None)
    if bv and bv > 0:
        ax.text(0.03, 0.97, f"β={bv:.1f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5, color=col, fontweight="bold")

def make_shared_legend(fig, avail):
    """Signal-level legend + arch color legend, placed below figure."""
    from matplotlib.lines import Line2D
    # signal strength patches
    sl_handles = [
        Line2D([0],[0], color="gray", alpha=a, linewidth=1.5,
               label=f"$s={sl:.1f}$" + (" (full)" if sl == 1.0 else ""))
        for sl, a in [(0.2, 0.25),(0.6, 0.55),(1.0, 1.0)]
    ]
    # dashed = sigmoid fit
    sl_handles.append(Line2D([0],[0], color="gray", alpha=0.75,
                              linewidth=0.9, linestyle="--", label="Sigmoid fit"))
    # architecture colours
    arch_handles = [
        mpatches.Patch(color=ARCH_COLOR[a], label=ARCH_LABEL[a])
        for a in ["FF","REC","SSM"]
    ]
    all_handles = sl_handles + arch_handles
    leg = fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(all_handles),
        fontsize=7,
        framealpha=0.9,
        edgecolor="0.7",
        title="Signal strength / Architecture",
        title_fontsize=7,
        handlelength=1.5,
        columnspacing=1.0,
        borderpad=0.5,
    )
    return leg

# ════════════════════════════════════════════════════════════════
# FIGURE A — Representative 3-panel (main body)
# 3 architecturally distinct models × 1 task (determiner-noun agreement)
# ════════════════════════════════════════════════════════════════
avail, task_keys = get_avail_and_tasks()
REP_MODELS = ["gemma2-2b", "huginn-3.5b", "mamba-2.8b"]
REP_LABELS = {
    "gemma2-2b":   r"Gemma 2 2B  (Feedforward, $\hat\beta=203.96$)",
    "huginn-3.5b": r"Huginn-3.5B  (Recurrent-depth, $\hat\beta=111.03$)",
    "mamba-2.8b":  r"Mamba-2.8B  (SSM, $\hat\beta=78.87$)",
}
TASK_REP = "blimp_determiner_noun_agreement_1"

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5),
                         gridspec_kw={"wspace": 0.28})
for i, (ax, mk) in enumerate(zip(axes, REP_MODELS)):
    plot_model_task(ax, mk, TASK_REP,
                    show_xlabel=True,
                    show_ylabel=(i == 0),
                    small=True)
    ax.set_title(REP_LABELS[mk], fontsize=7.5, pad=4)

# NO suptitle — LaTeX caption handles it
make_shared_legend(fig, avail)
fig.tight_layout(rect=[0, 0.08, 1, 1])

for ext in ["pdf","png"]:
    out = OUTDIR / f"fig2_representative.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ════════════════════════════════════════════════════════════════
# FIGURE B — Full 12-model grid (appendix)
# Models as rows, tasks as columns — more compact than aggregate.py default
# ════════════════════════════════════════════════════════════════
n_m = len(avail)
n_t = len(task_keys)

# Short task labels for column headers
TASK_SHORT = {
    "blimp_regular_plural_subject_verb_agreement_1": "SVA (easy)",
    "blimp_determiner_noun_agreement_1":             "Det-Noun (easy)",
    "blimp_principle_A_c_command":                   "Reflexive C-cmd",
    "blimp_wh_island":                               "Wh-island",
    "blimp_npi_present_1":                           "NPI island",
    "conll_ner":                                     "CoNLL NER",
    "ud_ewt":                                        "UD dep. type",
}

fig, axes = plt.subplots(n_m, n_t,
                         figsize=(3.2 * n_t, 2.2 * n_m),
                         gridspec_kw={"wspace": 0.22, "hspace": 0.55},
                         squeeze=False)

for mi, mk in enumerate(avail):
    arch = MODEL_REGISTRY[mk][3]
    for ti, tk in enumerate(task_keys):
        ax = axes[mi][ti]
        show_y = (ti == 0)
        show_x = (mi == n_m - 1)
        plot_model_task(ax, mk, tk,
                        show_xlabel=show_x,
                        show_ylabel=show_y,
                        small=True)
        # Column header only on first row
        if mi == 0:
            ax.set_title(TASK_SHORT.get(tk, tk[:22]), fontsize=8, pad=4)
        # Row label (model name + arch) on right side of last column
        if ti == n_t - 1:
            col = ARCH_COLOR[arch]
            ax.text(1.04, 0.5, mk, transform=ax.transAxes,
                    ha="left", va="center", fontsize=7,
                    color=col, rotation=0, fontweight="bold")

# NO suptitle
make_shared_legend(fig, avail)
fig.tight_layout(rect=[0, 0.04, 1, 1])

for ext in ["pdf","png"]:
    out = OUTDIR / f"fig2_full_grid.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

print("All done.")
