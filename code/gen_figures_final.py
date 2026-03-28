"""
gen_figures_final.py — NeurIPS 2026 Ignition Index Figures

Generates ALL paper figures with NO embedded titles (LaTeX captions only):
  fig2_main_body.pdf     — 3-panel representative (main body)
  figB1_det_noun.pdf     — Appendix B.1: all 12 models, det-noun
  figB2_npi.pdf          — Appendix B.2: all 12 models, NPI
  figB3_c_command.pdf    — Appendix B.3: all 12 models, C-command
  fig4_arch_comparison.pdf — Architecture bar chart (no embedded title)
  fig5_signal_heatmap.pdf  — Signal heatmap (no embedded title)
  fig6_training_dynamics.pdf — Training dynamics (no embedded title)

Usage:
    cd $PROJECT/ignition_index/code
    python gen_figures_final.py
"""
import pickle, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config import MODEL_REGISTRY, PARAM_COUNTS, SIGNAL_LEVELS, RESULTS_DIR, FIGURES_DIR
from src.probing import fit_sigmoid, sigmoid_4p

# ── Global NeurIPS style ─────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize":7.5,
    "ytick.labelsize":7.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.3,
    "grid.alpha":     0.25,
    "lines.linewidth":1.5,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})

ARCH_COLOR = {"FF": "#1f77b4", "REC": "#d62728", "SSM": "#2ca02c"}
ARCH_LABEL = {"FF": "Feedforward", "REC": "Recurrent-depth", "SSM": "SSM (no attn)"}

MODEL_DISPLAY = {
    "gpt2-small":  "GPT-2 Small (124M)",
    "gpt2-medium": "GPT-2 Medium (355M)",
    "gpt2-xl":     "GPT-2 XL (1.5B)",
    "pythia-70m":  "Pythia-70M",
    "pythia-410m": "Pythia-410M",
    "pythia-1.4b": "Pythia-1.4B",
    "pythia-6.9b": "Pythia-6.9B",
    "gemma2-2b":   "Gemma 2 2B (2.6B)",
    "gemma2-9b":   "Gemma 2 9B (9.2B)",
    "huginn-3.5b": "Huginn-3.5B",
    "mamba-1.4b":  "Mamba-1.4B",
    "mamba-2.8b":  "Mamba-2.8B",
}

SIG_TYPE   = "S1"
PLOT_LEVELS = [(0.2, 0.25), (0.6, 0.55), (1.0, 1.00)]

MODEL_ORDER = [
    "gpt2-small","gpt2-medium","gpt2-xl",
    "pythia-70m","pythia-410m","pythia-1.4b","pythia-6.9b",
    "gemma2-2b","gemma2-9b",
    "huginn-3.5b",
    "mamba-1.4b","mamba-2.8b",
]

APPENDIX_TASKS = [
    ("figB1_det_noun",  "blimp_determiner_noun_agreement_1"),
    ("figB2_npi",       "blimp_npi_present_1"),
    ("figB3_c_command", "blimp_principle_A_c_command"),
]


# ── IO ───────────────────────────────────────────────────────────
def load_model(mk):
    safe = mk.replace("-","_").replace(".","p")
    for suf in ["","_partial"]:
        p = RESULTS_DIR / f"model_{safe}{suf}.pkl"
        if p.exists():
            return pickle.load(open(p,"rb"))
    return None

def get_avail():
    return [mk for mk in MODEL_ORDER if load_model(mk) is not None]

def load_pkl(name):
    p = RESULTS_DIR / f"{name}.pkl"
    return pickle.load(open(p,"rb")) if p.exists() else None

def savefig(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    for fmt in ["pdf","png"]:
        out = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(out, dpi=300 if fmt=="png" else None, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ── Shared legend ────────────────────────────────────────────────
def make_legend_handles():
    sig = [
        Line2D([0],[0],color="gray",alpha=0.25,linewidth=2.0,label="$s=0.2$"),
        Line2D([0],[0],color="gray",alpha=0.55,linewidth=2.0,label="$s=0.6$"),
        Line2D([0],[0],color="gray",alpha=1.00,linewidth=2.0,label="$s=1.0$"),
        Line2D([0],[0],color="gray",alpha=0.75,linewidth=1.2,
               linestyle="--",label="Sigmoid fit ($s{=}1.0$)"),
    ]
    arch = [mpatches.Patch(color=ARCH_COLOR[a],label=ARCH_LABEL[a])
            for a in ["FF","REC","SSM"]]
    return sig + arch

def add_bottom_legend(fig, rect_bottom=0.10):
    """Place legend below panels. rect_bottom controls reserved space."""
    handles = make_legend_handles()
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), fontsize=7.5,
               framealpha=0.95, edgecolor="0.75",
               handlelength=1.8, handletextpad=0.5,
               columnspacing=0.9, borderpad=0.6)


# ── Core panel plot ──────────────────────────────────────────────
def plot_panel(ax, mk, task_key, show_xlabel=True, show_ylabel=True,
               title_override=None, fontsize_title=9):
    data = load_model(mk)
    arch = MODEL_REGISTRY[mk][3]
    col  = ARCH_COLOR[arch]

    if data is None:
        ax.text(0.5,0.5,"no data",transform=ax.transAxes,
                ha="center",va="center",color="gray",fontsize=8)
    else:
        curves = data.get("acc_curves_disc",{})
        nl_actual = data.get("n_layers", 12)
        # Use markers for small models (< 10 layers) so sparse lines are visible
        use_markers = nl_actual < 10

        for sl, alpha_ in PLOT_LEVELS:
            key = (task_key, SIG_TYPE, sl)
            if key not in curves:
                continue
            acc = np.array(curves[key])
            n   = len(acc)
            x   = np.linspace(0, 1, n)
            lw  = 1.8 if sl == 1.0 else 1.1
            ax.plot(x, acc*100, color=col, alpha=alpha_, linewidth=lw,
                    marker="o" if use_markers else None,
                    markersize=4 if use_markers else None,
                    zorder=3)
            if sl == 1.0:
                try:
                    ft = fit_sigmoid(x, acc, return_all=True)
                    if ft.get("popt") is not None:
                        xf = np.linspace(0,1,300)
                        yf = sigmoid_4p(xf,*ft["popt"])*100
                        ax.plot(xf,yf,"--",color=col,alpha=0.75,
                                linewidth=1.0,zorder=2)
                        b = ft["beta"]
                        if b and b > 0:
                            ax.text(0.04,0.96,
                                    f"$\\hat{{\\beta}}={b:.1f}$",
                                    transform=ax.transAxes,
                                    ha="left",va="top",fontsize=6.5,
                                    color=col,fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.15",
                                              facecolor="white",alpha=0.75,
                                              edgecolor="none"))
                except Exception:
                    pass

        # Layer count note for small models
        if use_markers:
            ax.text(0.97,0.04,f"{nl_actual}L",
                    transform=ax.transAxes,ha="right",va="bottom",
                    fontsize=6,color="gray",style="italic")

    ax.set_xlim(-0.02,1.02)
    ax.set_ylim(42,103)
    ax.set_yticks([50,70,90])
    ax.set_xticks([0.0,0.5,1.0])
    ax.grid(True)
    ax.spines[["top","right"]].set_visible(False)

    if show_xlabel:
        ax.set_xlabel(r"Layer $\ell/L$")
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Probe acc. (%)")
    else:
        ax.set_yticklabels([])

    title = title_override if title_override else MODEL_DISPLAY.get(mk,mk)
    ax.set_title(title, fontsize=fontsize_title, pad=4, fontweight="bold")


# ════════════════════════════════════════════════════════════════
# FIG 2 — Main body 3-panel representative
# ════════════════════════════════════════════════════════════════
def make_fig2_main():
    REP = ["gemma2-2b","huginn-3.5b","mamba-2.8b"]
    TASK = "blimp_determiner_noun_agreement_1"
    TITLES = {
        "gemma2-2b":   "Gemma 2 2B\n"
                       r"(Feedforward, $\hat\beta=203.96$)",
        "huginn-3.5b": "Huginn-3.5B\n"
                       r"(Recurrent-depth, $\hat\beta=111.03$)",
        "mamba-2.8b":  "Mamba-2.8B\n"
                       r"(SSM, $\hat\beta=78.87$)",
    }
    fig, axes = plt.subplots(1,3,figsize=(7.0,2.7),
                             gridspec_kw={"wspace":0.30})
    for i,(ax,mk) in enumerate(zip(axes,REP)):
        plot_panel(ax,mk,TASK,show_xlabel=True,
                   show_ylabel=(i==0),
                   title_override=TITLES[mk],fontsize_title=8.5)
    fig.tight_layout(rect=[0,0.13,1,1])
    add_bottom_legend(fig)
    savefig(fig,"fig2_main_body")


# ════════════════════════════════════════════════════════════════
# FIG B1–B3 — Appendix full grids
# ════════════════════════════════════════════════════════════════
def make_appendix_figure(fname, task_key):
    avail  = get_avail()
    models = [mk for mk in MODEL_ORDER if mk in avail]
    n      = len(models)
    ncols  = 3
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows,ncols,
                             figsize=(9.5, nrows*2.8),
                             gridspec_kw={"wspace":0.28,"hspace":0.58},
                             squeeze=False)
    flat = axes.flatten()

    for i,mk in enumerate(models):
        row,col = divmod(i,ncols)
        show_x  = (i >= n - ncols) or (row == nrows-1)
        show_y  = (col == 0)
        plot_panel(flat[i],mk,task_key,
                   show_xlabel=show_x,show_ylabel=show_y,
                   fontsize_title=9)

    for j in range(n,len(flat)):
        flat[j].set_visible(False)

    fig.tight_layout(rect=[0,0.07,1,1])
    add_bottom_legend(fig)
    savefig(fig,fname)


# ════════════════════════════════════════════════════════════════
# FIG 4 — Architecture comparison bar chart (NO embedded title)
# ════════════════════════════════════════════════════════════════
def make_fig4():
    avail = get_avail()
    sig   = "S1"

    def agg_beta(mk):
        r = load_model(mk)
        if r is None: return np.nan
        betas = [v for (t,st,sl),v in r.get("beta_hat",{}).items()
                 if st==sig and v>0]
        return float(np.mean(betas)) if betas else np.nan

    fig, ax = plt.subplots(figsize=(7.5,4.0))
    grp = defaultdict(list)
    for mk in avail:
        grp[MODEL_REGISTRY[mk][3]].append(mk)

    xi = 0
    xtick_pos, xtick_labels = [], []
    for arch in ["FF","REC","SSM"]:
        if arch not in grp: continue
        for mk in sorted(grp[arch], key=lambda m: agg_beta(m)):
            b  = agg_beta(mk)
            if np.isnan(b): continue
            r  = load_model(mk)
            ci = r.get("ci",{})
            ci_vals = [(lo,hi) for (_,st,_),(lo,hi) in ci.items()
                       if st==sig and not any(np.isnan(x) for x in (lo,hi))]
            lo_m = np.mean([c[0] for c in ci_vals]) if ci_vals else b-5
            hi_m = np.mean([c[1] for c in ci_vals]) if ci_vals else b+5
            ax.bar(xi,b,color=ARCH_COLOR[arch],alpha=0.85,edgecolor="white",width=0.7)
            ax.errorbar(xi,b,yerr=[[b-lo_m],[hi_m-b]],
                        fmt="none",color="black",capsize=3.5,linewidth=1.2)
            short = mk.replace("pythia-","p").replace("gpt2-","g2")\
                      .replace("gemma2-","g2-").replace("huginn-","hug-")\
                      .replace("mamba-","mb-")
            xtick_pos.append(xi)
            xtick_labels.append(short)
            xi += 1
        xi += 0.6   # gap between arch groups

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"Ignition Index $\overline{\hat{\beta}}$")
    ax.set_ylim(0, None)
    ax.grid(axis="y",alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    # Arch colour legend inside plot (upper left)
    ax.legend(handles=[mpatches.Patch(color=ARCH_COLOR[a],label=ARCH_LABEL[a])
                        for a in ["FF","REC","SSM"] if a in grp],
              loc="upper left", fontsize=8, framealpha=0.9)
    # NO ax.set_title or fig.suptitle
    fig.tight_layout()
    savefig(fig,"fig4_architecture_comparison")


# ════════════════════════════════════════════════════════════════
# FIG 5 — Signal heatmap (NO embedded title)
# ════════════════════════════════════════════════════════════════
def make_fig5():
    avail = get_avail()
    sig   = "S1"
    tk0   = "blimp_determiner_noun_agreement_1"

    mat = np.full((len(avail),len(SIGNAL_LEVELS)),np.nan)
    for mi,mk in enumerate(avail):
        r = load_model(mk)
        if r is None: continue
        for si,sl in enumerate(SIGNAL_LEVELS):
            b = r["beta_hat"].get((tk0,sig,sl),np.nan)
            mat[mi,si] = b if isinstance(b,float) and b>0 else np.nan

    fig,ax = plt.subplots(figsize=(6.5, max(2.5,0.62*len(avail))))
    im = ax.imshow(mat,aspect="auto",cmap="YlOrRd",vmin=0,vmax=300)
    ax.set_xticks(range(len(SIGNAL_LEVELS)))
    ax.set_xticklabels([f"$s={s:.1f}$" for s in SIGNAL_LEVELS])
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels([MODEL_DISPLAY.get(mk,mk) for mk in avail], fontsize=8)
    ax.set_xlabel("Input signal strength $s$")
    cb = plt.colorbar(im,ax=ax,shrink=0.8)
    cb.set_label(r"$\hat{\beta}$")
    # Annotate cells with value
    for mi in range(len(avail)):
        for si in range(len(SIGNAL_LEVELS)):
            v = mat[mi,si]
            if not np.isnan(v):
                ax.text(si,mi,f"{v:.0f}",ha="center",va="center",
                        fontsize=6,color="black" if v<200 else "white")
    # NO suptitle
    fig.tight_layout()
    savefig(fig,"fig5_signal_heatmap")


# ════════════════════════════════════════════════════════════════
# FIG 6 — Training dynamics (NO embedded title)
# ════════════════════════════════════════════════════════════════
def make_fig6():
    td = load_pkl("TRAINING_RESULTS")
    if td is None:
        print("  TRAINING_RESULTS.pkl not found — skipping fig6")
        return

    slugs   = sorted(td.keys())
    n       = len(slugs)
    fig,axes = plt.subplots(1,n,figsize=(6.5*n,4.2),squeeze=False)

    for pi,slug in enumerate(slugs):
        ax   = axes[0][pi]
        info = td[slug]
        steps = np.array(info["steps"])
        betas = np.array(info["betas"])
        valid = ~np.isnan(betas)

        ax.plot(steps[valid],betas[valid],"o-",color="#1f77b4",
                linewidth=1.5,markersize=4,label=r"$\hat{\beta}$")

        for cp in info.get("changepoints",[]):
            ax.axvline(cp,color="#d62728",linestyle="--",linewidth=1.5,
                       label=f"PELT cp={int(cp)}")
            pre  = betas[valid & (steps<cp)]
            post = betas[valid & (steps>=cp)]
            if len(pre) and len(post):
                ax.axhline(pre.mean(), color="#d62728",linestyle=":",
                           alpha=0.5,linewidth=1.0)
                ax.axhline(post.mean(),color="#2ca02c",linestyle=":",
                           alpha=0.5,linewidth=1.0)
                ax.text(0.03,0.96,
                        f"pre$={pre.mean():.2f}$\npost$={post.mean():.2f}$",
                        transform=ax.transAxes,va="top",fontsize=8,
                        bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.8))

        ax.set_xscale("symlog",linthresh=512)
        ax.set_xlabel("Training step")
        ax.set_ylabel(r"$\hat{\beta}$ (Ignition Index)")
        # Panel title = model name only (not figure caption)
        ax.set_title(info.get("model",slug), fontsize=9, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8,loc="lower right")

    # NO suptitle
    fig.tight_layout()
    savefig(fig,"fig6_training_dynamics")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Available models:", get_avail(), "\n")

    print("--- Fig 2: Main body representative ---")
    make_fig2_main()

    for fname,task_key in APPENDIX_TASKS:
        print(f"\n--- {fname} ---")
        make_appendix_figure(fname,task_key)

    print("\n--- Fig 4: Architecture comparison ---")
    make_fig4()

    print("\n--- Fig 5: Signal heatmap ---")
    make_fig5()

    print("\n--- Fig 6: Training dynamics ---")
    make_fig6()

    print("\nAll figures done.")
