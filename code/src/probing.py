"""
probing.py — Probe training, sigmoid fitting, BCa bootstrap,
              Extra-SS F-test, DeltaAICc.
Paper Sec 4.3–4.4.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, f as f_dist, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from src.config import (PROBE_CV_FOLDS, PROBE_MAX_ITER, PROBE_C_GRID,
                        N_BOOTSTRAP, ALPHA, AICC_THRESHOLD, N_PERM)

# ── Probe ────────────────────────────────────────────────────────
def train_probe(X_train, y_train, X_val=None, y_val=None, C_grid=None):
    if C_grid is None: C_grid = PROBE_C_GRID
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train.astype(np.float32))
    if X_val is None:
        best_C, best_score = C_grid[0], -np.inf
        for C in C_grid:
            clf = LogisticRegression(C=C, max_iter=PROBE_MAX_ITER, random_state=42, n_jobs=-1)
            sc  = cross_val_score(clf, X_tr, y_train, cv=PROBE_CV_FOLDS, scoring="accuracy").mean()
            if sc > best_score: best_score, best_C = sc, C
        clf = LogisticRegression(C=best_C, max_iter=PROBE_MAX_ITER, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_train)
        return best_score, best_score, clf
    X_v    = scaler.transform(X_val.astype(np.float32))
    best_C, best_score = C_grid[0], -np.inf
    for C in C_grid:
        clf = LogisticRegression(C=C, max_iter=PROBE_MAX_ITER, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_train)
        sc  = clf.score(X_v, y_val)
        if sc > best_score: best_score, best_C = sc, C
    clf = LogisticRegression(C=best_C, max_iter=PROBE_MAX_ITER, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)
    acc_disc = clf.score(X_v, y_val)
    proba    = clf.predict_proba(X_v)
    classes  = list(clf.classes_)
    log_liks = [np.log(proba[i, classes.index(yi) if yi in classes else 0] + 1e-12)
                for i, yi in enumerate(y_val)]
    return acc_disc, float(np.mean(log_liks)), clf

# ── 4-parameter sigmoid (Eq. 1) ──────────────────────────────────
def sigmoid_4p(x, y_min, y_max, x0, beta):
    return y_min + (y_max - y_min) / (1.0 + np.exp(-beta * (x - x0)))
def fit_sigmoid(layers_norm, accs, return_all=False):
    x   = np.asarray(layers_norm, dtype=np.float64)
    acc = np.asarray(accs,        dtype=np.float64)
    # Flatness check: if curve has no variation, return beta=0
    if np.std(acc) < 0.02 or (np.max(acc) - np.min(acc)) < 0.04:
        d = {"beta": 0.0, "x0": 0.5, "y_min": acc.min(), "y_max": acc.max(),
             "r2": 0.0, "popt": None, "rss": float(np.sum((acc - acc.mean())**2)), "n": len(acc)}
        return d if return_all else (0.0, 0.5, acc.min(), acc.max(), 0.0)
    y_min0 = np.percentile(acc,  5)
    y_max0 = np.percentile(acc, 95)
    mid    = (y_min0 + y_max0) / 2.0
    cross  = np.where(np.diff(np.sign(acc - mid)))[0]
    x0_0   = x[cross[0]] if len(cross) else 0.5
    beta0  = max(0.5, 4.0 * (acc[-1] - acc[0]) / max(y_max0 - y_min0, 1e-6))
    p0     = [y_min0, y_max0, x0_0, beta0]
    bounds = ([0.0, 0.0, -0.5, 0.0], [1.0, 1.0, 1.5, 300.0])
    try:
        popt, pcov = curve_fit(sigmoid_4p, x, acc, p0=p0, bounds=bounds,
                               maxfev=20000, method="trf")
        y_pred = sigmoid_4p(x, *popt)
        ss_res = np.sum((acc - y_pred) ** 2)
        ss_tot = np.sum((acc - acc.mean()) ** 2) + 1e-10
        r2     = 1.0 - ss_res / ss_tot
        d      = {"beta": popt[3], "x0": popt[2], "y_min": popt[0],
                  "y_max": popt[1], "r2": r2, "popt": popt,
                  "rss": float(ss_res), "n": len(acc)}
        return d if return_all else (popt[3], popt[2], popt[0], popt[1], r2)
    except Exception:
        d = {"beta": 0.0, "x0": 0.5, "y_min": acc.min(), "y_max": acc.max(),
             "r2": 0.0, "popt": None, "rss": float(np.sum((acc - acc.mean())**2)), "n": len(acc)}
        return d if return_all else (0.0, 0.5, acc.min(), acc.max(), 0.0)

def transition_width_layers(beta, n_layers):
    if beta <= 0: return float(n_layers)
    return (np.log(81.0) / beta) * n_layers

# ── BCa bootstrap ────────────────────────────────────────────────
def bca_ci(layers_norm, accs, B=None, alpha=None):
    if B is None:     B     = N_BOOTSTRAP
    if alpha is None: alpha = ALPHA
    x, acc = np.asarray(layers_norm), np.asarray(accs)
    n      = len(acc)
    theta  = fit_sigmoid(x, acc)[0]
    rng    = np.random.RandomState(42)
    boots  = np.array([fit_sigmoid(x[i:=rng.randint(0,n,n)], acc[i])[0] for _ in range(B)])
    z0     = norm.ppf(np.clip(np.mean(boots < theta), 1e-6, 1-1e-6))
    jack   = np.array([fit_sigmoid(np.delete(x,i), np.delete(acc,i))[0] for i in range(n)])
    jm     = jack.mean()
    a      = np.sum((jm-jack)**3) / (6.0 * (np.sum((jm-jack)**2)**1.5 + 1e-10))
    def adj(za): return norm.cdf(z0 + (z0 + za) / (1.0 - a*(z0 + za))) * 100
    lo = np.percentile(boots, np.clip(adj(norm.ppf(alpha/2)),   0.1, 99.9))
    hi = np.percentile(boots, np.clip(adj(norm.ppf(1-alpha/2)), 0.1, 99.9))
    return theta, lo, hi

# ── Extra-SS F-test (Motulsky 2004) — H1/H2 ──────────────────────
def extra_ss_ftest(layers_a, accs_a, layers_b, accs_b):
    xa, ya = np.asarray(layers_a), np.asarray(accs_a)
    xb, yb = np.asarray(layers_b), np.asarray(accs_b)
    n      = len(xa) + len(xb)
    fit_a  = fit_sigmoid(xa, ya, return_all=True)
    fit_b  = fit_sigmoid(xb, yb, return_all=True)
    rss_sep = fit_a["rss"] + fit_b["rss"]
    df_sep  = n - 6

    def shared_model(X, ym_a, yM_a, x0_a, beta, ym_b, yM_b, x0_b):
        xa_, xb_ = X[:len(ya)], X[len(ya):]
        ya_ = ym_a + (yM_a - ym_a) / (1.0 + np.exp(-beta * (xa_ - x0_a)))
        yb_ = ym_b + (yM_b - ym_b) / (1.0 + np.exp(-beta * (xb_ - x0_b)))
        return np.concatenate([ya_, yb_])

    x_pool = np.concatenate([xa, xb])
    y_pool = np.concatenate([ya, yb])
    p0_sh  = [fit_a["y_min"], fit_a["y_max"], fit_a["x0"],
              (fit_a["beta"] + fit_b["beta"]) / 2.0,
              fit_b["y_min"], fit_b["y_max"], fit_b["x0"]]
    bounds_sh = ([0,0,-0.5,  0, 0,0,-0.5], [1,1, 1.5,300, 1,1, 1.5])
    try:
        popt_sh, _ = curve_fit(shared_model, x_pool, y_pool,
                               p0=p0_sh, bounds=bounds_sh, maxfev=30000)
        rss_shared = float(np.sum((y_pool - shared_model(x_pool, *popt_sh))**2))
        df_shared  = n - 7
    except Exception:
        rss_shared = rss_sep * 1.1
        df_shared  = n - 7

    df_extra = df_sep - df_shared
    if df_extra <= 0 or df_sep <= 0: return np.nan, np.nan, fit_a["beta"], fit_b["beta"]
    F = ((rss_shared - rss_sep) / df_extra) / (rss_sep / df_sep + 1e-10)
    p = float(1.0 - f_dist.cdf(max(F, 0), df_extra, df_sep))
    return float(F), p, float(fit_a["beta"]), float(fit_b["beta"])

# ── DeltaAICc (Schaeffer 2023) ────────────────────────────────────
def aicc(n, k, rss):
    aic = n * np.log(rss / n + 1e-12) + 2 * k
    return aic + 2 * k * (k + 1) / max(n - k - 1, 1)

def delta_aicc(layers_norm, accs):
    x, y = np.asarray(layers_norm), np.asarray(accs)
    n    = len(y)
    c    = np.polyfit(x, y, 1)
    rss_lin = float(np.sum((y - np.polyval(c, x))**2))
    fit     = fit_sigmoid(x, y, return_all=True)
    rss_sig = fit["rss"] if fit["popt"] is not None else rss_lin
    best_step = np.inf
    for t in range(1, n-1):
        pred = np.concatenate([np.full(t, y[:t].mean()), np.full(n-t, y[t:].mean())])
        rss  = float(np.sum((y - pred)**2))
        if rss < best_step: best_step = rss
    return {
        "delta_linear": aicc(n,2,rss_lin) - aicc(n,4,rss_sig),
        "delta_step":   aicc(n,2,best_step) - aicc(n,4,rss_sig),
        "schaeffer_flag": (aicc(n,2,best_step) - aicc(n,4,rss_sig)) < AICC_THRESHOLD,
    }

# ── Permutation Spearman (H5) ─────────────────────────────────────
def permutation_spearman_p(x, y, n_perm=None):
    if n_perm is None: n_perm = N_PERM
    x, y    = np.asarray(x), np.asarray(y)
    rho_obs, _ = spearmanr(x, y)
    rng     = np.random.RandomState(0)
    null    = [spearmanr(rng.permutation(x), y)[0] for _ in range(n_perm)]
    p_upper = np.mean(np.array(null) >= rho_obs)
    return float(rho_obs), float(2 * min(p_upper, 1 - p_upper))
