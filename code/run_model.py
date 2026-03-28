"""
run_model.py — Single-model experiment runner.
Called by SLURM array jobs. Each job handles exactly one model.

Usage:
    python run_model.py --model gpt2-small --signal_types S1 S2 S3
    python run_model.py --model pythia-6.9b --signal_types S1
"""
import argparse
import gc
import logging
import pickle
import sys
import time
from pathlib import Path
import os

import numpy as np
import torch

# ── add project root to path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    MODEL_REGISTRY, SIGNAL_LEVELS, RESULTS_DIR, LOGS_DIR,
    N_PROBE_SAMPLES, PROBE_C_GRID,
)
from src.datasets import load_all_datasets
from src.signals import (
    apply_s1_masking, apply_s3_pos_corruption,
    EmbNoiseHook, embedding_sigma,
)
from src.probing import (
    train_probe, fit_sigmoid, bca_ci,
    transition_width_layers, delta_aicc,
)

# ── Logging ──────────────────────────────────────────────────────
def setup_logger(model_key):
    log_file = LOGS_DIR / f"{model_key}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)

# ── Save / load ───────────────────────────────────────────────────
def save_pkl(obj, name):
    path = RESULTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path

def load_pkl(name):
    path = RESULTS_DIR / f"{name}.pkl"
    return pickle.load(open(path, "rb")) if path.exists() else None

# ── Activation extraction ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_tl(hf_id, sentences, s, sig_type, batch_size=8, max_len=128, model=None):
    from transformer_lens import HookedTransformer
    _owns_model = model is None
    if _owns_model:
        model = HookedTransformer.from_pretrained(hf_id, device=DEVICE,
                  center_unembed=True, center_writing_weights=True,
                  fold_ln=True, local_files_only=True)
    m   = model
    m.eval()
    tok = m.tokenizer; tok.pad_token = tok.eos_token
    n_layers = m.cfg.n_layers; d_model = m.cfg.d_model
    emb_sig  = embedding_sigma(m, s=0.0, is_tl=True)
    acts     = {l: [] for l in range(n_layers)}
    rng      = np.random.RandomState(42)
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc   = tok(batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
        ids   = enc["input_ids"].to(DEVICE)
        amask = enc["attention_mask"].to(DEVICE)
        last  = (amask.sum(-1) - 1).to(DEVICE)
        if sig_type == "S1":
            ids = apply_s1_masking(ids, s, tok.pad_token_id or 0, rng)
        elif sig_type == "S3":
            ids  = apply_s3_pos_corruption(batch, s, tok, DEVICE, rng)
            last = (ids != (tok.pad_token_id or 0)).sum(-1) - 1
        noise_sigma = (1.0-s)*emb_sig if sig_type == "S2" else 0.0
        with torch.no_grad():
            with EmbNoiseHook(m, noise_sigma, is_tl=True):
                _, cache = m.run_with_cache(
                    ids, names_filter=lambda n: "resid_pre" in n, return_type=None)
        for l in range(n_layers):
            hs = cache["resid_pre", l]
            lh = hs[torch.arange(len(batch)), last, :]
            acts[l].append(lh.float().cpu().numpy())
    for l in range(n_layers):
        acts[l] = np.concatenate(acts[l], axis=0)
    if _owns_model:
        del m; torch.cuda.empty_cache(); gc.collect()
    return acts, {"n_layers": n_layers, "d_model": d_model}

def extract_huginn(sentences, s, sig_type, batch_size=4, max_len=128, n_iters=32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "tomg-group-umd/huginn-0125"
    hm  = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
        local_files_only=True)
    hm.eval()
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.eos_token
    d_model = hm.config.hidden_size
    rng     = np.random.RandomState(42)
    acts    = {i: [] for i in range(n_iters)}

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc   = tok(batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
        ids   = enc["input_ids"].to(DEVICE)
        amask = enc["attention_mask"].to(DEVICE)
        last  = (amask.sum(-1) - 1).to(DEVICE)
        if sig_type == "S1":
            ids = apply_s1_masking(ids, s, tok.pad_token_id or 0, rng)

        # Use hooks to capture hidden states at each recurrent iteration
        iter_hiddens = []
        hooks = []
        def make_hook(it_list):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                it_list.append(h.detach().float().cpu())
            return hook_fn

        # Hook LAST layer of core_block only — fires once per recurrent pass
        iter_hiddens.clear()
        core_block = hm.transformer.core_block
        hooks.append(core_block[-1].register_forward_hook(make_hook(iter_hiddens)))

        with torch.no_grad():
            hm(input_ids=ids, attention_mask=amask)

        for hk in hooks:
            hk.remove()

        n_actual = min(n_iters, len(iter_hiddens))
        for it, hid in enumerate(iter_hiddens[:n_actual]):
            lh = hid[torch.arange(len(batch)), last.cpu().long(), :]
            acts[it].append(lh.numpy())
        # pad remaining iters with last hidden state
        if iter_hiddens:
            last_hid = iter_hiddens[-1]
            for it in range(n_actual, n_iters):
                lh = last_hid[torch.arange(len(batch)), last.cpu().long(), :]
                acts[it].append(lh.numpy())

    for it in range(n_iters):
        acts[it] = np.concatenate(acts[it], axis=0) if acts[it] \
                   else np.zeros((len(sentences), d_model), np.float32)
    del hm; torch.cuda.empty_cache(); gc.collect()
    return acts, {"n_layers": n_iters, "d_model": d_model}

def extract_mamba(hf_id, sentences, s, sig_type, batch_size=4, max_len=128):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hm  = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    hm.eval()
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.pad_token = tok.eos_token
    n_layers = hm.config.num_hidden_layers; d_model = hm.config.hidden_size
    rng      = np.random.RandomState(42)
    acts     = {l: [] for l in range(n_layers)}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc   = tok(batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
        ids   = enc["input_ids"].to(DEVICE)
        amask = enc["attention_mask"].to(DEVICE)
        last  = (amask.sum(-1) - 1).to(DEVICE)
        if sig_type == "S1":
            ids = apply_s1_masking(ids, s, tok.pad_token_id or 0, rng)
        with torch.no_grad():
            out = hm(input_ids=ids, output_hidden_states=True)
        for l, hs in enumerate(out.hidden_states[:n_layers]):
            lh = hs[torch.arange(len(batch)), last, :]
            acts[l].append(lh.float().cpu().numpy())
    for l in range(n_layers):
        acts[l] = np.concatenate(acts[l], axis=0) if acts[l] \
                  else np.zeros((len(sentences), d_model), np.float32)
    del hm; torch.cuda.empty_cache(); gc.collect()
    return acts, {"n_layers": n_layers, "d_model": d_model}

# ── Core experiment ───────────────────────────────────────────────
def run_model(model_key, signal_types, datasets, logger, shuffled=False):
    safe_key = model_key.replace("-","_").replace(".","p")
    suffix = "_shuffled" if shuffled else ""
    suffix = "_shuffled" if shuffled else ""
    cached = load_pkl(f"model_{safe_key}{suffix}")
    if cached is not None:
        # Only use cache if all requested signal types are present
        cached_levels = set(map(tuple, cached.get("_completed_levels", [])))
        requested = set((st, sl) for st in signal_types
                        for sl in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if requested.issubset(cached_levels):
            logger.info(f"[CACHE HIT] {model_key} — loading from disk")
            return cached
        else:
            logger.info(f"[CACHE INCOMPLETE] {model_key} — missing signal types, rerunning")
            # Move to partial so resume logic picks it up
            import shutil
            results_dir = Path(os.environ["SCRATCH"]) / "ignition_index" / "results"
            src_path = results_dir / f"model_{safe_key}.pkl"
            dst_path = results_dir / f"model_{safe_key}_partial.pkl"
            shutil.move(str(src_path), str(dst_path))

    # Resume from partial checkpoint if available
    partial = load_pkl(f"model_{safe_key}{suffix}_partial")
    completed_levels = set()
    if partial is not None:
        logger.info(f"[RESUME] {model_key} — resuming from partial checkpoint")
        res = partial
        completed_levels = set(map(tuple, partial.get("_completed_levels", [])))
    else:
        partial = None

    hf_id, _, _, arch, use_tl = MODEL_REGISTRY[model_key]
    logger.info(f"Starting: {model_key}  arch={arch}  hf_id={hf_id}")

    probe_datasets = {k: v for k, v in datasets.items()
                      if v.get("task") in ("T1","T2","T3")}

    # Collect all unique sentences
    all_sents, seen = [], set()
    for td in probe_datasets.values():
        for s in td["sentences"]:
            if s not in seen: seen.add(s); all_sents.append(s)

    if partial is None:
        res = {
            "model_key": model_key, "arch": arch, "hf_id": hf_id,
            "acc_curves_disc": {}, "acc_curves_loglik": {},
            "beta_hat": {}, "ci": {}, "delta_aicc": {},
            "transition_width": {}, "r2": {},
            "n_layers": None, "d_model": None,
            "_completed_levels": [],
        }

    # Load TL model once outside loop to avoid repeated Hub calls
    _tl_model = None
    if use_tl:
        from transformer_lens import HookedTransformer
        logger.info(f"  Loading {model_key} into HookedTransformer (once)...")
        _tl_model = HookedTransformer.from_pretrained(
            hf_id, device=DEVICE,
            center_unembed=True, center_writing_weights=True,
            fold_ln=True, local_files_only=True,
            dtype='bfloat16')
        _tl_model.eval()
        logger.info(f"  Model loaded. n_layers={_tl_model.cfg.n_layers}")


    try:
        for stype in signal_types:
            for slevel in SIGNAL_LEVELS:
                if (stype, slevel) in completed_levels:
                    logger.info(f"  [SKIP] signal type={stype}  s={slevel:.1f} (already done)")
                    continue
                t0 = time.time()
                logger.info(f"  signal type={stype}  s={slevel:.1f}")
                try:
                    if use_tl:
                        bs = 16 if "70m" in model_key else (8 if "410m" in model_key else 4)
                        acts, cfg = extract_tl(hf_id, all_sents, s=slevel,
                                               sig_type=stype, batch_size=bs, model=_tl_model)
                    elif arch == "REC":
                        acts, cfg = extract_huginn(all_sents, s=slevel,
                                                   sig_type=stype, batch_size=2)
                    elif arch == "SSM":
                        acts, cfg = extract_mamba(hf_id, all_sents, s=slevel,
                                                  sig_type=stype, batch_size=2)
                    else:
                        logger.warning(f"  Unknown arch {arch} — skip")
                        continue
                except Exception as e:
                    logger.error(f"  Extraction FAILED: {e}")
                    import traceback; traceback.print_exc()
                    continue

                n_layers = cfg["n_layers"]
                res["n_layers"] = n_layers
                res["d_model"]  = cfg["d_model"]
                lnorm = np.linspace(0, 1, n_layers)
                sent_idx = {s: i for i, s in enumerate(all_sents)}

                for tkey, tdata in probe_datasets.items():
                    sents  = tdata["sentences"]
                    labels = tdata["labels"]
                    if args.shuffled:
                        rng_shuffle = np.random.RandomState(42)
                        labels = rng_shuffle.permutation(labels)

                    sidx   = np.array([sent_idx.get(s, 0) for s in sents])
                    rng    = np.random.RandomState(42)
                    perm   = rng.permutation(len(sents)); sp = int(0.8 * len(sents))
                    ti, vi = perm[:sp], perm[sp:]

                    disc_accs, loglik_accs = [], []
                    for l in range(n_layers):
                        X = acts[l][sidx]
                        try:
                            da, la, _ = train_probe(X[ti], labels[ti], X[vi], labels[vi])
                        except Exception:
                            da, la = 0.5, np.log(0.5)
                        disc_accs.append(da)
                        loglik_accs.append(la)

                    key = (tkey, stype, slevel)
                    res["acc_curves_disc"][key]   = disc_accs
                    res["acc_curves_loglik"][key] = loglik_accs
                    fit = fit_sigmoid(lnorm, disc_accs, return_all=True)
                    res["beta_hat"][key]         = fit["beta"]
                    res["r2"][key]               = fit["r2"]
                    res["transition_width"][key] = transition_width_layers(fit["beta"], n_layers)
                    res["delta_aicc"][key]       = delta_aicc(lnorm, disc_accs)

                    if stype == "S1":
                        bh, lo, hi = bca_ci(lnorm, disc_accs)
                        res["ci"][key] = (lo, hi)
                    else:
                        res["ci"][key] = (np.nan, np.nan)

                    flag = "!" if res["delta_aicc"][key]["schaeffer_flag"] else " "
                    logger.info(
                        f"    {tkey:35s} beta={fit['beta']:6.2f}  "
                        f"w={res['transition_width'][key]:5.1f}L  "
                        f"R2={fit['r2']:.3f} {flag}"
                    )

                dt = time.time() - t0
                logger.info(f"  Done s={slevel:.1f}  ({dt/60:.1f} min)")
                res.setdefault("_completed_levels", []).append((stype, slevel))
                suffix = "_shuffled" if args.shuffled else ""
                save_pkl(res, f"model_{safe_key}{suffix}_partial")
                logger.info(f"  [CHECKPOINT] Partial saved after s={slevel:.1f}")

    finally:
        if _tl_model is not None:
            del _tl_model; torch.cuda.empty_cache(); gc.collect()

    save_pkl(res, f"model_{safe_key}{suffix}")
    logger.info(f"DONE {model_key}. Saved model_{safe_key}.pkl")
    return res


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model key from MODEL_REGISTRY")
    parser.add_argument("--signal_types", nargs="+", default=["S1"],
                        choices=["S1","S2","S3"],
                        help="Signal manipulation types to run")
    parser.add_argument("--n_samples", type=int, default=None)

    parser.add_argument("--shuffled", action="store_true",
                        help="Run shuffled-label control")
    args = parser.parse_args()

    logger = setup_logger(args.model)
    logger.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    logger.info("Loading datasets...")
    datasets = load_all_datasets(n=args.n_samples)
    logger.info(f"Datasets: {list(datasets.keys())}")

    run_model(args.model, args.signal_types, datasets, logger, args.shuffled)
