"""
run_training_dynamics.py — H4: Pythia checkpoints + PELT changepoint.
Called by jobs/run_training_dynamics.sh

FIX (Mar 2026): Load checkpoints fully offline by reading cached ref hashes
from disk and using local_files_only=True. HookedTransformer.from_pretrained
with checkpoint_value= hits the Hub even with TRANSFORMERS_OFFLINE=1.
"""
import gc, logging, os, pickle, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RESULTS_DIR, LOGS_DIR, PYTHIA_CHECKPOINTS, PELT_PENALTY
from src.datasets import load_all_datasets
from src.probing import fit_sigmoid, train_probe
import ruptures as rpt

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "training_dynamics.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_HOME = Path(os.environ.get("HF_HOME",
    Path(__file__).parent.parent / "hf_cache"))


def load_pkl(name):
    p = RESULTS_DIR / f"{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def save_pkl(obj, name):
    with open(RESULTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def _resolve_revision(model_id: str, step: int) -> str:
    folder = "models--" + model_id.replace("/", "--")
    refs_dir = HF_HOME / "hub" / folder / "refs"
    ref_file = refs_dir / f"step{step}"
    if ref_file.exists():
        return ref_file.read_text().strip()
    if step == 143000:
        main_file = refs_dir / "main"
        if main_file.exists():
            return main_file.read_text().strip()
    raise FileNotFoundError(
        f"No cached ref for {model_id} step {step}. Expected: {ref_file}"
    )


def _load_hookedtransformer_offline(model_id: str, step: int):
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    revision = _resolve_revision(model_id, step)
    log.info(f"  Resolved step {step} -> revision {revision[:12]}...")

    folder       = "models--" + model_id.replace("/", "--")
    snapshot_dir = HF_HOME / "hub" / folder / "snapshots" / revision

    # Tokenizer from main snapshot (step 143000 = main branch)
    main_rev  = _resolve_revision(model_id, 143000)
    tok_dir   = HF_HOME / "hub" / folder / "snapshots" / main_rev

    # Pass LOCAL PATHS directly — bypasses ALL HF hub/cache lookup logic
    tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)

    hf_model = GPTNeoXForCausalLM.from_pretrained(
        str(snapshot_dir),          # <-- directory path, not model_id
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    m = HookedTransformer.from_pretrained(
        model_id, hf_model=hf_model, tokenizer=tok, device=DEVICE,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
    )

    del hf_model
    torch.cuda.empty_cache()
    gc.collect()
    return m



def run_one_model(model_id, model_slug, sents, labs, ti, vi):
    from transformer_lens import HookedTransformer
    TRAIN_RES = {}

    for step in tqdm(PYTHIA_CHECKPOINTS, desc=f"{model_slug} checkpoints"):
        ck_key = f"training_dyn_{model_slug}_step{step}"
        cached = load_pkl(ck_key)
        if cached is not None:
            TRAIN_RES[step] = cached
            log.info(f"  [CACHE] {model_slug} step {step}: beta={cached:.4f}")
            continue

        log.info(f"  Loading {model_slug} checkpoint step={step}...")
        try:
            m = _load_hookedtransformer_offline(model_id, step)
            m.eval()
            tok = m.tokenizer
            tok.pad_token = tok.eos_token
            n_layers = m.cfg.n_layers

            enc = tok(sents, return_tensors="pt", padding=True,
                      truncation=True, max_length=128)
            ids_ = enc["input_ids"].to(DEVICE)
            last_ = enc["attention_mask"].sum(-1) - 1

            with torch.no_grad():
                _, cache = m.run_with_cache(
                    ids_,
                    names_filter=lambda n: "resid_pre" in n,
                    return_type=None,
                )

            accs = []
            for l in range(n_layers):
                X = cache["resid_pre", l][
                    torch.arange(len(sents)), last_, :
                ].float().cpu().numpy()
                try:
                    a, _, _ = train_probe(X[ti], labs[ti], X[vi], labs[vi])
                except Exception:
                    a = 0.5
                accs.append(a)

            beta = fit_sigmoid(np.linspace(0, 1, n_layers), accs)[0]
            TRAIN_RES[step] = beta
            save_pkl(beta, ck_key)
            log.info(f"  {model_slug} step {step:8d}: beta={beta:.4f}")

            del m, cache
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            log.info(f"  FAIL {model_slug} step {step}: {e}")
            import traceback; traceback.print_exc()
            TRAIN_RES[step] = np.nan

    return TRAIN_RES


def run_pelt(train_res):
    steps_arr = np.array(sorted(train_res))
    betas_arr = np.array([train_res[s] for s in steps_arr])
    valid = ~np.isnan(betas_arr)

    changepoints = []
    if valid.sum() >= 5:
        sig_ = betas_arr[valid].reshape(-1, 1)
        cps = rpt.Pelt(model="rbf", min_size=2).fit(sig_).predict(pen=PELT_PENALTY)
        changepoints = steps_arr[valid][np.array(cps[:-1], dtype=int) - 1].tolist()
        log.info(f"  PELT changepoints: {changepoints}")
    else:
        log.info("  Not enough valid points for PELT")

    return {
        "betas_by_step": train_res,
        "changepoints": changepoints,
        "steps": steps_arr.tolist(),
        "betas": betas_arr.tolist(),
    }


def run():
    log.info("Loading datasets...")
    datasets = load_all_datasets(n=200)
    tkey = [k for k, v in datasets.items() if v.get("task") == "T1"][0]
    sents = datasets[tkey]["sentences"][:200]
    labs  = datasets[tkey]["labels"][:200]
    rng   = np.random.RandomState(42)
    perm  = rng.permutation(len(sents))
    sp    = int(0.8 * len(sents))
    ti, vi = perm[:sp], perm[sp:]

    log.info(f"Task: {tkey}, N={len(sents)}, device={DEVICE}")

    MODELS = [
        ("EleutherAI/pythia-410m", "pythia410m"),
        ("EleutherAI/pythia-1.4b", "pythia1p4b"),
    ]

    all_results = {}

    for model_id, model_slug in MODELS:
        log.info(f"\n{'='*50}")
        log.info(f"Running training dynamics: {model_id}")
        log.info(f"{'='*50}")

        train_res = run_one_model(model_id, model_slug, sents, labs, ti, vi)
        result    = run_pelt(train_res)
        result["model"] = model_id

        all_results[model_slug] = result
        save_pkl(result, f"TRAINING_RESULTS_{model_slug}")
        log.info(f"Saved TRAINING_RESULTS_{model_slug}.pkl")

    save_pkl(all_results, "TRAINING_RESULTS")
    log.info("Training dynamics complete. Saved TRAINING_RESULTS.pkl")


if __name__ == "__main__":
    run()
