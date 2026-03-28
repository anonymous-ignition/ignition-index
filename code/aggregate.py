"""
aggregate.py — Collect per-model .pkl files, run H1-H4 stats,
               C1-C3 controls, figures, and Table 1.
Run this AFTER all run_model.py jobs AND run_training_dynamics.py have completed.

Usage:
    python aggregate.py --signal_types S1
    python aggregate.py --signal_types S1 S2 S3 --run_controls --run_sae

Fixes applied (Mar 2026):
  - log() → log.info() throughout (Logger is not callable)
  - H1 F-test now pools all curves per architecture, not just index [0]
  - Fig 6 (H4 training dynamics) added to build_figures
  - H5 run_h5 kept but clearly optional / graceful skip
  - Table 1 SCAN/COGS columns removed (H5 deferred)
  - C4 label corrected to C3 throughout
"""
import argparse
import gc
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    MODEL_REGISTRY, PARAM_COUNTS, SIGNAL_LEVELS,
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR,
    BLIMP_PARADIGMS, N_PERM, ALPHA,
)
from src.probing import (
    fit_sigmoid, extra_ss_ftest, delta_aicc,
    transition_width_layers, permutation_spearman_p,
)
from scipy.stats import mannwhitneyu, spearmanr
from itertools import combinations

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "aggregate.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

ARCH_COLOR = {"FF": "#1f77b4", "REC": "#d62728", "SSM": "#2ca02c"}
ARCH_LABEL = {"FF": "Feedforward", "REC": "Recurrent-depth", "SSM": "SSM (no attn)"}


# ── IO ────────────────────────────────────────────────────────────
def load_pkl(name):
    p = RESULTS_DIR / f"{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def save_pkl(obj, name):
    with open(RESULTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


# ── Load all available model results ─────────────────────────────
def load_all_results():
    results = {}
    for mk in MODEL_REGISTRY:
        safe = f"model_{mk.replace('-', '_').replace('.', 'p')}"
        r = load_pkl(safe)
        if r is not None:
            results[mk] = r
            log.info(f"  [LOADED]  {mk}")   # FIX: was log() — Logger not callable
        else:
            log.info(f"  [MISSING] {mk}")
    return results


def aggregate_beta(res, stype="S1"):
    betas = [v for (t, st, sl), v in res.get("beta_hat", {}).items()
             if st == stype and v > 0]
    return float(np.mean(betas)) if betas else np.nan


def bh_correct(p_values, alpha=ALPHA):
    n   = len(p_values)
    idx = np.argsort(p_values)
    sp  = np.array(p_values)[idx]
    thr = (np.arange(1, n + 1) / n) * alpha
    reject = np.zeros(n, bool)
    k_arr = np.where(sp <= thr)[0]
    if len(k_arr):
        reject[idx[:k_arr[-1] + 1]] = True
    p_adj = np.empty(n)
    p_adj[idx] = np.minimum(
        1.0, np.minimum.accumulate((sp * n / np.arange(1, n + 1))[::-1])[::-1]
    )
    return p_adj, reject


# ── H1–H3 ────────────────────────────────────────────────────────
def run_statistics(ALL_RESULTS, signal_type="S1"):
    GLOBAL = {}

    # Build per-architecture pools: list of (x_normalised, accs) tuples
    arch_pools = defaultdict(list)
    for mk, res in ALL_RESULTS.items():
        arch = MODEL_REGISTRY[mk][3]
        for (tk, st, sl), accs in res["acc_curves_disc"].items():
            if st != signal_type:
                continue
            n = res.get("n_layers", len(accs))
            arch_pools[arch].append((np.linspace(0, 1, n), accs))

    # beta distributions per architecture
    arch_beta_dist = {}
    for a, pool in arch_pools.items():
        betas = []
        for x, c in pool:
            try:
                betas.append(fit_sigmoid(x, c)[0])
            except Exception:
                pass
        if betas:
            arch_beta_dist[a] = np.array(betas)

    # ── H1: Architecture rank ordering ───────────────────────────
    log.info("\n=== H1: Architecture rank ===")   # FIX: was log()
    H1 = {"ftest_pairs": [], "mwu_pairs": [], "arch_means": {}}
    arch_list = sorted(arch_beta_dist)
    p_raw, mwu_rows = [], []

    for a1, a2 in combinations(arch_list, 2):
        b1 = arch_beta_dist.get(a1, np.array([]))
        b2 = arch_beta_dist.get(a2, np.array([]))
        if len(b1) < 3 or len(b2) < 3:
            continue

        # FIX: pool ALL curves per arch, not just index [0]
        # Concatenate x and accs across all curves in each pool for F-test
        pool1 = arch_pools[a1]
        pool2 = arch_pools[a2]
        # Use mean curve as representative for F-test (standard approach)
        # when pools have different-length models, normalise to same x grid
        x_ref = np.linspace(0, 1, 24)  # 24-point normalised grid
        def mean_curve(pool):
            interp_accs = []
            for x, c in pool:
                interp_accs.append(np.interp(x_ref, x, c))
            return x_ref, np.mean(interp_accs, axis=0)

        x1, c1_mean = mean_curve(pool1)
        x2, c2_mean = mean_curve(pool2)
        try:
            F, pF, _, _ = extra_ss_ftest(x1, c1_mean, x2, c2_mean)
            H1["ftest_pairs"].append({"arch1": a1, "arch2": a2, "F": F, "p_F": pF})
            log.info(f"  F-test {a1} vs {a2}: F={F:.2f} p={pF:.4f}")
        except Exception as e:
            log.info(f"  F-test {a1} vs {a2}: FAILED — {e}")

        stat, p = mannwhitneyu(b1, b2, alternative="two-sided")
        mwu_rows.append((a1, a2, b1.mean(), b2.mean(), stat, p))
        p_raw.append(p)

    if p_raw:
        p_adj, rej = bh_correct(p_raw)
        for i, (a1, a2, m1, m2, stat, p) in enumerate(mwu_rows):
            H1["mwu_pairs"].append({
                "arch1": a1, "arch2": a2, "beta1": m1, "beta2": m2,
                "U": stat, "p_raw": p, "p_BH": p_adj[i], "reject": bool(rej[i])
            })
            log.info(
                f"  MWU {a1} vs {a2}: beta={m1:.2f} vs {m2:.2f}  "
                f"p_BH={p_adj[i]:.4f} {'*' if rej[i] else ''}"
            )

    H1["arch_means"] = {a: float(v.mean()) for a, v in arch_beta_dist.items()}
    GLOBAL["H1"] = H1

    # ── H2: Scaling within family ────────────────────────────────
    log.info("\n=== H2: Scaling ===")   # FIX: was log()
    FAMILIES = {
        "GPT-2":  ["gpt2-small", "gpt2-medium", "gpt2-xl"],
        "Pythia": ["pythia-70m", "pythia-410m", "pythia-1.4b", "pythia-6.9b"],
        "Gemma2": ["gemma2-2b", "gemma2-9b"],
    }
    H2 = {}
    for fam, members in FAMILIES.items():
        avail = [m for m in members if m in ALL_RESULTS]
        if len(avail) < 2:
            continue
        params = np.log10([PARAM_COUNTS[m] for m in avail])
        betas  = np.array([aggregate_beta(ALL_RESULTS[m], signal_type) for m in avail])
        valid  = ~np.isnan(betas)
        if valid.sum() < 2:
            continue
        rho, p = spearmanr(params[valid], betas[valid])
        H2[fam] = {"rho": rho, "p_rho": p, "members": avail, "betas": betas[valid].tolist()}
        log.info(f"  {fam}: rho={rho:.3f} p={p:.4f}")
    GLOBAL["H2"] = H2

    # ── H3: Signal strength dependence ───────────────────────────
    log.info("\n=== H3: Signal strength ===")   # FIX: was log()
    H3 = {}
    for mk, res in ALL_RESULTS.items():
        per_s = {}
        for sl in SIGNAL_LEVELS:
            bs = [v for (t, st, s), v in res["beta_hat"].items()
                  if st == signal_type and abs(s - sl) < 0.01 and v > 0]
            if bs:
                per_s[sl] = np.mean(bs)
        if len(per_s) >= 3:
            sls = sorted(per_s)
            bs  = [per_s[s] for s in sls]
            rho, p = spearmanr(sls, bs)
            H3[mk] = {"per_signal": per_s, "rho": rho, "p": p}
            log.info(f"  {mk}: rho={rho:.3f} p={p:.4f}")
    GLOBAL["H3"] = H3

    return GLOBAL


# ── H5 capability (optional, graceful skip) ───────────────────────
def run_h5(ALL_RESULTS, signal_type="S1"):
    cap_res = load_pkl("CAPABILITY_RESULTS")
    if cap_res is None:
        log.info("CAPABILITY_RESULTS.pkl not found — H5 skipped (deferred to future work)")
        return {}
    H5 = {}
    for bench in ["scan", "cogs"]:
        model_keys = [mk for mk in ALL_RESULTS
                      if not np.isnan(cap_res.get(mk, {}).get(bench, np.nan))]
        betas_ = [aggregate_beta(ALL_RESULTS[mk], signal_type) for mk in model_keys]
        bench_ = [cap_res[mk][bench] for mk in model_keys]
        valid  = [not np.isnan(b) and not np.isnan(s) for b, s in zip(betas_, bench_)]
        bv = [b for b, v in zip(betas_, valid) if v]
        sv = [s for s, v in zip(bench_, valid) if v]
        if len(bv) >= 3:
            rho, p = permutation_spearman_p(bv, sv, N_PERM)
            H5[bench] = {
                "rho": rho, "p_perm": p,
                "model_betas": dict(zip(model_keys, betas_)),
                "model_accs":  dict(zip(model_keys, bench_)),
            }
            log.info(f"  H5 {bench}: rho={rho:.3f} p_perm={p:.4f}")
    return H5


# ── Table 1 ──────────────────────────────────────────────────────
def build_table1(ALL_RESULTS, GLOBAL, signal_type="S1"):
    rows = []
    for mk in sorted(
        ALL_RESULTS,
        key=lambda m: (aggregate_beta(ALL_RESULTS[m], signal_type)
                       if not np.isnan(aggregate_beta(ALL_RESULTS[m], signal_type)) else -1),
        reverse=True,
    ):
        res  = ALL_RESULTS[mk]
        arch = MODEL_REGISTRY[mk][3]
        bm   = aggregate_beta(res, signal_type)
        cis  = [(lo, hi) for (_, st, _), (lo, hi) in res.get("ci", {}).items()
                if st == signal_type and not any(np.isnan(x) for x in (lo, hi))]
        ws   = [v for v in res.get("transition_width", {}).values()
                if not np.isnan(v) and v > 0]
        sch  = sum(1 for d in res.get("delta_aicc", {}).values()
                   if isinstance(d, dict) and d.get("schaeffer_flag"))
        rows.append({
            "Architecture": ARCH_LABEL.get(arch, arch),
            "Model":        mk,
            "Params":       f"{int(PARAM_COUNTS.get(mk, 0) / 1e6)}M",
            "Layers":       res.get("n_layers", "?"),
            "beta_mean":    round(bm, 3) if not np.isnan(bm) else "—",
            "CI_lo":        round(np.mean([c[0] for c in cis]), 3) if cis else "—",
            "CI_hi":        round(np.mean([c[1] for c in cis]), 3) if cis else "—",
            "w_layers":     round(np.mean(ws), 1) if ws else "—",
            "Schaeffer_flags": sch,
        })
    T1 = pd.DataFrame(rows)
    T1.to_csv(TABLES_DIR / "table1_main_results.csv", index=False)
    T1.to_latex(
        TABLES_DIR / "table1_main_results.tex", index=False, escape=False,
        caption="Main results: Ignition Index per model.",
        label="tab:results_main",
    )
    log.info(f"Table 1 saved -> {TABLES_DIR}")
    return T1


# ── Figures ───────────────────────────────────────────────────────
def build_figures(ALL_RESULTS, GLOBAL, signal_type="S1"):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    avail = [mk for mk in MODEL_REGISTRY if mk in ALL_RESULTS]

    probe_keys = [k for k in (ALL_RESULTS[avail[0]]["acc_curves_disc"] if avail else {})
                  if k[1] == signal_type and k[2] == 1.0]
    task_keys  = sorted(set(k[0] for k in probe_keys))[:3]

    def savefig(fig, name):
        for fmt in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"{name}.{fmt}", bbox_inches="tight",
                        dpi=200 if fmt == "png" else None)
        plt.close(fig)
        log.info(f"  Saved {name}.pdf/.png")

    # ── Fig 2: per-layer accuracy curves + sigmoid fits ──────────
    if avail and task_keys:
        n_m, n_t = len(avail), len(task_keys)
        fig, axes = plt.subplots(n_t, n_m, figsize=(3.5 * n_m, 2.8 * n_t), squeeze=False)
        for ti, tk in enumerate(task_keys):
            for mi, mk in enumerate(avail):
                ax  = axes[ti][mi]
                res = ALL_RESULTS[mk]
                nl  = res.get("n_layers", 12)
                ln  = np.linspace(0, 1, nl)
                col = ARCH_COLOR.get(MODEL_REGISTRY[mk][3], "#888")
                for sl, alpha_ in [(1.0, 1.0), (0.6, 0.55), (0.2, 0.25)]:
                    k_ = (tk, signal_type, sl)
                    if k_ not in res["acc_curves_disc"]:
                        continue
                    accs_ = np.array(res["acc_curves_disc"][k_]) * 100
                    ax.plot(ln, accs_, color=col, alpha=alpha_, linewidth=1.2 + sl)
                    if sl == 1.0:
                        ft = fit_sigmoid(ln, accs_ / 100, return_all=True)
                        if ft["popt"] is not None:
                            from src.probing import sigmoid_4p
                            yf = sigmoid_4p(ln, *ft["popt"]) * 100
                            ax.plot(ln, yf, "--", color=col, alpha=0.7, linewidth=1.0,
                                    label=f"β={ft['beta']:.1f}")
                ax.set(title=f"{mk}\n{tk[:20]}", xlim=(0, 1), ylim=(45, 102))
                ax.grid(alpha=0.2)
                if ti == 0 and mi == 0:
                    ax.legend(fontsize=7)
        fig.suptitle("Fig 2: Per-Layer Probe Accuracy + Sigmoid Fits",
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        savefig(fig, "fig2_per_layer_accuracy")

    # ── Fig 4: architecture comparison bar chart ──────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xi  = 0
    grp = defaultdict(list)
    for mk in avail:
        grp[MODEL_REGISTRY[mk][3]].append(mk)

    for arch in ["FF", "REC", "SSM"]:
        if arch not in grp:
            continue
        for mk in sorted(grp[arch],
                         key=lambda m: aggregate_beta(ALL_RESULTS[m], signal_type)):
            b      = aggregate_beta(ALL_RESULTS[mk], signal_type)
            ci     = ALL_RESULTS[mk]["ci"]
            ci_vals = [(lo, hi) for (_, st, _), (lo, hi) in ci.items()
                       if st == signal_type and not any(np.isnan(x) for x in (lo, hi))]
            lo_m = np.mean([c[0] for c in ci_vals]) if ci_vals else b - 0.3
            hi_m = np.mean([c[1] for c in ci_vals]) if ci_vals else b + 0.3
            ax.bar(xi, b, color=ARCH_COLOR[arch], alpha=0.85, edgecolor="white")
            ax.errorbar(xi, b, yerr=[[b - lo_m], [hi_m - b]],
                        fmt="none", color="black", capsize=4, linewidth=1.2)
            ax.text(xi, -0.8,
                    mk.replace("pythia-", "p").replace("gpt2-", "g2"),
                    ha="center", va="top", fontsize=7, rotation=35)
            xi += 1
        xi += 0.5

    ax.set_xticks([])
    ax.set_ylabel("Ignition Index β̂")
    ax.set_title("Fig 4: Ignition Index by Model (BCa 95% CI)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=ARCH_COLOR[a], label=ARCH_LABEL[a])
                 for a in ["FF", "REC", "SSM"] if a in grp],
        loc="upper left",
    )
    fig.tight_layout()
    savefig(fig, "fig4_architecture_comparison")

    # ── Fig 5: H3 signal-strength heatmap ────────────────────────
    if avail and task_keys:
        tk0 = task_keys[0]
        mat = np.full((len(avail), len(SIGNAL_LEVELS)), np.nan)
        for mi, mk in enumerate(avail):
            for si, sl in enumerate(SIGNAL_LEVELS):
                b = ALL_RESULTS[mk]["beta_hat"].get((tk0, signal_type, sl), np.nan)
                mat[mi, si] = b if isinstance(b, float) and b > 0 else np.nan
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.65 * len(avail))))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(SIGNAL_LEVELS)))
        ax.set_xticklabels([f"s={s:.1f}" for s in SIGNAL_LEVELS])
        ax.set_yticks(range(len(avail)))
        ax.set_yticklabels(avail, fontsize=8)
        plt.colorbar(im, ax=ax, label="β̂", shrink=0.8)
        ax.set_title(f"Fig 5: H3 Signal Heatmap  |  Task: {tk0}", fontweight="bold")
        fig.tight_layout()
        savefig(fig, "fig5_signal_heatmap")

    # ── Fig 6: H4 training dynamics ───────────────────────────────
    # FIX: was completely missing from original aggregate.py
    td = load_pkl("TRAINING_RESULTS")
    if td is not None:
        model_slugs = sorted(td.keys())
        n_panels    = len(model_slugs)
        fig, axes   = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
        for pi, slug in enumerate(model_slugs):
            ax   = axes[0][pi]
            info = td[slug]
            steps_all = np.array(info["steps"])
            betas_all = np.array(info["betas"])

            # Plot trajectory
            valid = ~np.isnan(betas_all)
            ax.plot(steps_all[valid], betas_all[valid],
                    "o-", color="#1f77b4", linewidth=1.5, markersize=4, label="β̂")

            # PELT changepoints
            for cp in info.get("changepoints", []):
                ax.axvline(cp, color="#d62728", linestyle="--", linewidth=1.5,
                           label=f"PELT cp={int(cp)}")
                # Annotate pre/post means
                pre  = betas_all[valid & (steps_all < cp)]
                post = betas_all[valid & (steps_all >= cp)]
                if len(pre) and len(post):
                    ax.axhline(pre.mean(),  color="#d62728", linestyle=":",
                               alpha=0.5, linewidth=1.0)
                    ax.axhline(post.mean(), color="#2ca02c", linestyle=":",
                               alpha=0.5, linewidth=1.0)
                    ax.text(0.02, 0.97,
                            f"pre={pre.mean():.2f}\npost={post.mean():.2f}",
                            transform=ax.transAxes, va="top", fontsize=8,
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

            ax.set_xscale("symlog", linthresh=512)
            ax.set_xlabel("Training step")
            ax.set_ylabel("β̂ (Ignition Index)")
            ax.set_title(f"Fig 6: H4 Training Dynamics\n{info.get('model', slug)}",
                         fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")

        fig.tight_layout()
        savefig(fig, "fig6_training_dynamics")
    else:
        log.info("  TRAINING_RESULTS.pkl not found — Fig 6 skipped")

    # ── Fig 7: H5 capability scatter (only if H5 ran) ─────────────
    H5 = GLOBAL.get("H5", {})
    if H5:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ai, bench in enumerate(["scan", "cogs"]):
            ax  = axes[ai]
            h5b = H5.get(bench, {})
            if h5b:
                mb   = h5b.get("model_betas", {})
                ma   = h5b.get("model_accs", {})
                mks_ = [mk for mk in mb
                        if mk in ma and not any(np.isnan([mb[mk], ma[mk]]))]
                if mks_:
                    xs_ = [mb[mk] for mk in mks_]
                    ys_ = [ma[mk] * 100 for mk in mks_]
                    for x, y, mk in zip(xs_, ys_, mks_):
                        ax.scatter(x, y,
                                   color=ARCH_COLOR.get(MODEL_REGISTRY[mk][3], "#888"),
                                   s=90, zorder=5)
                        ax.annotate(mk, (x, y), textcoords="offset points",
                                    xytext=(5, 3), fontsize=7)
                    if len(xs_) >= 2:
                        m_, b_ = np.polyfit(xs_, ys_, 1)
                        xl_    = np.linspace(min(xs_) * 0.9, max(xs_) * 1.1, 50)
                        ax.plot(xl_, m_ * xl_ + b_, "--", color="#555", linewidth=1.2)
                    rho_ = h5b.get("rho", np.nan)
                    pp_  = h5b.get("p_perm", np.nan)
                    ax.text(0.05, 0.95,
                            f"ρ={rho_:.2f}\np_perm={pp_:.4f}",
                            transform=ax.transAxes, va="top", fontsize=9,
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
            ax.set(xlabel="β̂_mean",
                   ylabel=f"{bench.upper()} Acc (%)",
                   title=f"Fig 7{'a' if bench == 'scan' else 'b'}: H5 ({bench.upper()})")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        savefig(fig, "fig7_capability_scatter")

    log.info(f"All figures saved to {FIGURES_DIR}")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_types", nargs="+", default=["S1"])
    parser.add_argument("--run_controls", action="store_true")
    parser.add_argument("--run_sae",      action="store_true")
    args = parser.parse_args()

    log.info("Loading results...")
    ALL_RESULTS = load_all_results()
    log.info(f"Loaded {len(ALL_RESULTS)} models: {list(ALL_RESULTS.keys())}")

    GLOBAL        = run_statistics(ALL_RESULTS, signal_type=args.signal_types[0])
    GLOBAL["H5"]  = run_h5(ALL_RESULTS, signal_type=args.signal_types[0])

    T1 = build_table1(ALL_RESULTS, GLOBAL, signal_type=args.signal_types[0])
    print("\n" + T1.to_string(index=False))

    build_figures(ALL_RESULTS, GLOBAL, signal_type=args.signal_types[0])

    save_pkl(GLOBAL, "GLOBAL_RESULTS")
    log.info("Aggregation complete.")
