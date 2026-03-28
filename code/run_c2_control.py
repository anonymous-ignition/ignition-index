"""
C2 control: uses s=0.0 (fully masked input) curves as shuffled-label proxy.
At s=0.0, probe accuracy is at chance — equivalent to shuffled labels.
"""
import pickle, sys, os, numpy as np, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config import MODEL_REGISTRY, RESULTS_DIR
from src.probing import fit_sigmoid

def load_model(mk):
    safe = mk.replace("-","_").replace(".","p")
    for suf in ["","_partial"]:
        p = RESULTS_DIR / f"model_{safe}{suf}.pkl"
        if p.exists(): return pickle.load(open(p,"rb"))
    return None

null_betas = []
task_betas = []

for mk in MODEL_REGISTRY:
    r = load_model(mk)
    if r is None: continue
    curves = r.get("acc_curves_disc", {})

    for key, acc in curves.items():
        task, st, sl = key
        if st != "S1": continue

        acc = np.array(acc, dtype=float)
        if len(acc) < 4: continue
        x = np.linspace(0, 1, len(acc))

        # Null distribution: s=0.0 (fully masked = chance performance)
        if sl == 0.0:
            try:
                ft = fit_sigmoid(x, acc, return_all=True)
                b = ft.get("beta")
                if b and b > 0:
                    null_betas.append(min(b, 300))
            except: pass

        # Task: s=1.0 (full signal)
        if sl == 1.0:
            try:
                ft = fit_sigmoid(x, acc, return_all=True)
                b = ft.get("beta")
                if b and b > 0:
                    task_betas.append(min(b, 300))
            except: pass

null_betas = np.array(null_betas)
task_betas = np.array(task_betas)

print(f"Null (s=0.0): n={len(null_betas)}, mean={null_betas.mean():.2f}, "
      f"SD={null_betas.std():.2f}, median={np.median(null_betas):.2f}")
print(f"Task (s=1.0): n={len(task_betas)}, mean={task_betas.mean():.2f}, "
      f"SD={task_betas.std():.2f}, median={np.median(task_betas):.2f}")

# Selectivity
print(f"\nSelectivity (task - null mean): {task_betas.mean() - null_betas.mean():.2f}")
print(f"Task mean / null mean ratio: {task_betas.mean() / null_betas.mean():.1f}x")

result = {
    "null_mean": float(null_betas.mean()), "null_sd": float(null_betas.std()),
    "null_n": int(len(null_betas)),
    "task_mean": float(task_betas.mean()), "task_sd": float(task_betas.std()),
    "task_n": int(len(task_betas)),
    "selectivity": float(task_betas.mean() - null_betas.mean()),
}
json.dump(result, open(RESULTS_DIR/"c2_control.json","w"), indent=2)
print("\nSaved c2_control.json")
