import re

path = "/home/AUTHOR/projects/ACCOUNT/AUTHOR/ignition_index/code/run_model.py"
with open(path) as f:
    src = f.read()

old_cache = '''    cached   = load_pkl(f"model_{safe_key}")
    if cached is not None:
        logger.info(f"[CACHE HIT] {model_key} — loading from disk")
        return cached'''

new_cache = '''    cached = load_pkl(f"model_{safe_key}")
    if cached is not None:
        logger.info(f"[CACHE HIT] {model_key} — loading from disk")
        return cached

    # Resume from partial checkpoint if available
    partial = load_pkl(f"model_{safe_key}_partial")
    completed_levels = set()
    if partial is not None:
        logger.info(f"[RESUME] {model_key} — resuming from partial checkpoint")
        res = partial
        completed_levels = set(map(tuple, partial.get("_completed_levels", [])))
    else:
        partial = None'''

old_res_init = '''    res = {
        "model_key": model_key, "arch": arch, "hf_id": hf_id,
        "acc_curves_disc": {}, "acc_curves_loglik": {},
        "beta_hat": {}, "ci": {}, "delta_aicc": {},
        "transition_width": {}, "r2": {},
        "n_layers": None, "d_model": None,
    }'''

new_res_init = '''    if partial is None:
        res = {
            "model_key": model_key, "arch": arch, "hf_id": hf_id,
            "acc_curves_disc": {}, "acc_curves_loglik": {},
            "beta_hat": {}, "ci": {}, "delta_aicc": {},
            "transition_width": {}, "r2": {},
            "n_layers": None, "d_model": None,
            "_completed_levels": [],
        }'''

old_stype_loop = '''    for stype in signal_types:
        for slevel in SIGNAL_LEVELS:
            t0 = time.time()
            logger.info(f"  signal type={stype}  s={slevel:.1f}")'''

new_stype_loop = '''    for stype in signal_types:
        for slevel in SIGNAL_LEVELS:
            if (stype, slevel) in completed_levels:
                logger.info(f"  [SKIP] signal type={stype}  s={slevel:.1f} (already done)")
                continue
            t0 = time.time()
            logger.info(f"  signal type={stype}  s={slevel:.1f}")'''

old_done_log = '''            dt = time.time() - t0
            logger.info(f"  Done s={slevel:.1f}  ({dt/60:.1f} min)")

    save_pkl(res, f"model_{safe_key}")'''

new_done_log = '''            dt = time.time() - t0
            logger.info(f"  Done s={slevel:.1f}  ({dt/60:.1f} min)")
            res.setdefault("_completed_levels", []).append((stype, slevel))
            save_pkl(res, f"model_{safe_key}_partial")
            logger.info(f"  [CHECKPOINT] Partial saved after s={slevel:.1f}")

    save_pkl(res, f"model_{safe_key}")'''

patches = [
    (old_cache,      new_cache),
    (old_res_init,   new_res_init),
    (old_stype_loop, new_stype_loop),
    (old_done_log,   new_done_log),
]

for old, new in patches:
    if old not in src:
        print(f"PATCH FAILED — string not found:\n{old[:80]}...")
        exit(1)
    src = src.replace(old, new, 1)
    print(f"OK: patched '{old[:60].strip()}'")

with open(path, "w") as f:
    f.write(src)

print("\nAll patches applied successfully.")
