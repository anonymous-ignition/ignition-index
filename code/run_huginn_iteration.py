"""
run_huginn_iteration.py — Huginn iteration-level probing (T2.1)

Tests whether Huginn's ignition operates along the recurrent iteration axis
rather than the depth axis. Fixes depth at layer 16, probes across iterations 1-64.

Usage:
    python run_huginn_iteration.py --signal_types S1
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    SIGNAL_LEVELS, RESULTS_DIR, LOGS_DIR,
    PROBE_C_GRID,
)
from src.datasets import load_all_datasets
from src.signals import apply_s1_masking
from src.probing import train_probe, fit_sigmoid, bca_ci, transition_width_layers

# ── Setup ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_DEPTH = -1  # Use last layer (Huginn architecture)

def setup_logger():
    log_file = LOGS_DIR / "huginn_iteration.log"
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

def save_pkl(obj, name):
    path = RESULTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path

# ── Huginn iteration extraction ───────────────────────────────────
def extract_huginn_iterations(sentences, s, sig_type, batch_size=4, max_len=128, max_iters=64):
    """
    Extract Huginn activations across recurrent iterations at FIXED depth layer.
    
    Returns:
        acts: dict {iteration_num: array[n_samples, d_model]}
        meta: dict with model metadata
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    mid = "tomg-group-umd/huginn-0125"
    hm = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16, trust_remote_code=True, 
        device_map="auto", local_files_only=True)
    hm.eval()
    
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.eos_token
    
    d_model = hm.config.hidden_size
    rng = np.random.RandomState(42)
    
    # Storage: iteration -> list of batches
    acts = {i: [] for i in range(1, max_iters + 1)}
    
    for batch_idx in range(0, len(sentences), batch_size):
        batch = sentences[batch_idx:batch_idx + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        ids = enc["input_ids"].to(DEVICE)
        amask = enc["attention_mask"].to(DEVICE)
        last = (amask.sum(-1) - 1).to(DEVICE)
        
        if sig_type == "S1":
            ids = apply_s1_masking(ids, s, tok.pad_token_id or 0, rng)
        
        # Capture hidden states at each iteration
        iter_hiddens = []
        
        def make_hook(it_list):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                it_list.append(h.detach().float().cpu())
            return hook_fn
        
        # Hook the FIXED_DEPTH layer in core_block
        # Huginn's core_block is a ModuleList of transformer layers
        # Each iteration processes through all layers; we want layer FIXED_DEPTH's output each pass
        core_block = hm.transformer.core_block
        # Get actual index
        actual_depth = FIXED_DEPTH if FIXED_DEPTH >= 0 else len(core_block) + FIXED_DEPTH
        hook = core_block[actual_depth].register_forward_hook(make_hook(iter_hiddens))
        with torch.no_grad():
            hm(input_ids=ids, attention_mask=amask)
        
        hook.remove()
        
        # iter_hiddens now contains [iter1_output, iter2_output, ..., iter64_output]
        # Each is shape [batch_size, seq_len, d_model]
        for iter_num, hs in enumerate(iter_hiddens[:max_iters], start=1):
            # Extract final token position
            lh = hs[torch.arange(len(batch)), last.cpu(), :]
            acts[iter_num].append(lh.numpy())
    
    # Concatenate batches
    for iter_num in range(1, max_iters + 1):
        if acts[iter_num]:
            acts[iter_num] = np.concatenate(acts[iter_num], axis=0)
    
    del hm
    torch.cuda.empty_cache()
    
    return acts, {"d_model": d_model, "n_iters": max_iters, "fixed_depth": FIXED_DEPTH}

# ── Main experiment ────────────────────────────────────────────────
def run_huginn_iteration_experiment(signal_types):
    logger = setup_logger()
    logger.info("="*70)
    logger.info("HUGINN ITERATION-AXIS PROBING (T2.1)")
    logger.info(f"Fixed depth: layer {FIXED_DEPTH}")
    logger.info(f"Signal types: {signal_types}")
    logger.info("="*70)
    
    # Load datasets
    datasets = load_all_datasets()
    
    res = {
        "model_key": "huginn-iteration",
        "fixed_depth": FIXED_DEPTH,
        "acc_curves_disc": {},
        "beta_hat": {},
        "ci": {},
        "delta_aicc": {},
        "transition_width": {},
        "r2": {},
    }
    
    for stype in signal_types:
        logger.info(f"\n{'='*70}")
        logger.info(f"Signal type: {stype}")
        logger.info(f"{'='*70}")
        
        for slevel in SIGNAL_LEVELS:
            logger.info(f"\n  Signal level s={slevel:.1f}")
            
            for tkey, tdata in datasets.items():
                logger.info(f"    Task: {tkey}")
                
                sents = tdata["sentences"]
                labels = tdata["labels"]
                
                # Extract activations across iterations
                acts, meta = extract_huginn_iterations(sents, slevel, stype)
                n_iters = meta["n_iters"]
                
                # Create sentence index mapping
                sent_idx = {s: i for i, s in enumerate(sents)}
                sidx = np.array([sent_idx.get(s, 0) for s in sents])
                
                # Train/val split
                rng = np.random.RandomState(42)
                perm = rng.permutation(len(sents))
                sp = int(0.8 * len(sents))
                ti, vi = perm[:sp], perm[sp:]
                
                # Probe across iterations
                disc_accs = []
                for iter_num in range(1, n_iters + 1):
                    X = acts[iter_num]
                    try:
                        da, _, _ = train_probe(X[ti], labels[ti], X[vi], labels[vi])
                    except Exception:
                        da = 0.5
                    disc_accs.append(da)
                
                # Fit sigmoid over iteration axis
                inorm = np.arange(n_iters) / n_iters  # Normalized iteration [0,1]
                key = (tkey, stype, slevel)
                res["acc_curves_disc"][key] = disc_accs
                
                fit = fit_sigmoid(inorm, disc_accs, return_all=True)
                res["beta_hat"][key] = fit["beta"]
                res["ci"][key] = bca_ci(inorm, disc_accs, B=2000)
                res["delta_aicc"][key] = fit.get("delta_aicc", None)
                res["transition_width"][key] = transition_width_layers(fit["beta"], n_iters)
                res["r2"][key] = fit["r2"]
                
                logger.info(f"      β̂={fit['beta']:.1f}, R²={fit['r2']:.3f}")
    
    # Save results
    save_pkl(res, "huginn_iteration_probing")
    logger.info(f"\n{'='*70}")
    logger.info("SAVED: huginn_iteration_probing.pkl")
    logger.info(f"{'='*70}")
    
    return res

# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_types", nargs="+", default=["S1"],
                        choices=["S1", "S2", "S3"])
    args = parser.parse_args()
    
    run_huginn_iteration_experiment(args.signal_types)
