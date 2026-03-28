import pickle, sys, os, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from src.config import MODEL_REGISTRY, RESULTS_DIR, FIGURES_DIR
from src.probing import fit_sigmoid

FIGURES_DIR.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.family":"serif","font.size":9,
    "pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":0.6,"grid.alpha":0.25})

ARCH_COLOR = {"FF":"#1f77b4","REC":"#d62728","SSM":"#2ca02c"}

def load_model(mk):
    safe = mk.replace("-","_").replace(".","p")
    for suf in ["","_partial"]:
        p = RESULTS_DIR / f"model_{safe}{suf}.pkl"
        if p.exists(): return pickle.load(open(p,"rb"))
    return None

disc_betas, cont_betas, arch_labels = [], [], []

for mk in MODEL_REGISTRY:
    r = load_model(mk)
    if r is None: continue
    arch = MODEL_REGISTRY[mk][3]
    disc = r.get("acc_curves_disc", {})
    llik = r.get("acc_curves_loglik", {})
    shared_keys = set(disc.keys()) & set(llik.keys())
    print(f"  {mk}: {len(shared_keys)} shared keys")

    for key in shared_keys:
        d_acc = np.array(disc[key], dtype=float)
        c_ll  = np.array(llik[key], dtype=float)
        if len(d_acc) < 4: continue
        # Normalise loglik to [0,1]
        c_rng = c_ll.max() - c_ll.min()
        if c_rng < 1e-6: continue
        c_norm = (c_ll - c_ll.min()) / c_rng
        x = np.linspace(0, 1, len(d_acc))
        try:
            fd = fit_sigmoid(x, d_acc,  return_all=True)
            fc = fit_sigmoid(x, c_norm, return_all=True)
            bd = fd.get("beta"); bc = fc.get("beta")
            if bd and bc and bd > 0 and bc > 0:
                disc_betas.append(min(bd, 300))
                cont_betas.append(min(bc, 300))
                arch_labels.append(arch)
        except Exception:
            pass

disc_betas = np.array(disc_betas)
cont_betas = np.array(cont_betas)
print(f"\nC1: {len(disc_betas)} valid pairs")
if len(disc_betas) < 2:
    print("ERROR: not enough pairs"); sys.exit(1)

r_val, p_val = pearsonr(disc_betas, cont_betas)
print(f"Pearson r = {r_val:.3f}, p = {p_val:.4e}")

# Scatter plot
fig, ax = plt.subplots(figsize=(4.5, 4.0))
arch_display = {"FF":"Feedforward","REC":"Recurrent-depth","SSM":"SSM (no attn)"}
for arch in ["FF","REC","SSM"]:
    idx = [i for i,a in enumerate(arch_labels) if a==arch]
    if not idx: continue
    ax.scatter(disc_betas[idx], cont_betas[idx],
               color=ARCH_COLOR[arch], alpha=0.45, s=18,
               label=arch_display[arch], zorder=3)

lim = max(disc_betas.max(), cont_betas.max()) * 1.08
ax.plot([0,lim],[0,lim],"--",color="gray",linewidth=0.8,alpha=0.6,label="Identity")
m, b = np.polyfit(disc_betas, cont_betas, 1)
ax.plot(np.linspace(0,lim,200), m*np.linspace(0,lim,200)+b,
        "-", color="black", linewidth=1.2, alpha=0.8)
p_str = "$p < 0.001$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
ax.text(0.05, 0.93, f"$r = {r_val:.3f}$ ({p_str})",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.85,edgecolor="0.75"))
ax.set_xlabel(r"$\hat{\beta}$ — discrete accuracy")
ax.set_ylabel(r"$\hat{\beta}$ — continuous log-likelihood")
ax.set_xlim(0,lim); ax.set_ylim(0,lim); ax.set_aspect("equal")
ax.grid(True); ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
fig.tight_layout()
for ext in ["pdf","png"]:
    out = FIGURES_DIR / f"figC1_metric_comparison.{ext}"
    fig.savefig(out, dpi=300 if ext=="png" else None, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

json.dump({"n_pairs":int(len(disc_betas)),"pearson_r":float(r_val),
           "pearson_p":float(p_val)},
          open(FIGURES_DIR/"c1_summary.json","w"), indent=2)
print("Done.")
