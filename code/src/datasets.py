"""
datasets.py — Load and cache all datasets.

Matches paper specification exactly (Appendix B):
  T1: BLiMP grammatical acceptability
      - 5 paradigms defined in config.BLIMP_PARADIGMS
      - Binary (grammatical=1, ungrammatical=0), N=1000 minimal pairs each
      - HuggingFace: nyu-mll/blimp
  T2: CoNLL-2003 NER (binary entity-present=1 / absent=0)
      - English test split, N=3,684, natural class distribution (no balancing)
      - Loaded from local parquet (download_data.sh)
  T3: Universal Dependencies EN-EWT v2.13
      - 10-way syntactic dependency classification
      - Train split, N=12,543 sentences
      - Loaded from local .conllu (download_data.sh)
  H5A: SCAN add-primitive (jump) — skipped, unavailable as parquet on HF Hub
  H5B: COGS compositional generalisation
       - gen split, up to 21,000 examples
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from src.config import BLIMP_PARADIGMS

# ── Data paths ─────────────────────────────────────────────────────────────────
# PROJECT env var is set in SLURM scripts; falls back to hardcoded Narval path.
_DATA = os.path.join(
    os.environ.get("PROJECT",
                   "/home/AUTHOR/projects/ACCOUNT/AUTHOR"),
    "ignition_index", "data"
)
_CONLL_DIR = os.path.join(_DATA, "conll2003")
_UD_DIR    = os.path.join(_DATA, "ud_ewt")

# ── UD dependency → class map (paper Table A2, 10 classes) ────────────────────
UD_DEP_MAP = {
    "nsubj":      0,   # nominal subject
    "obj":        1,   # object
    "iobj":       2,   # indirect object
    "csubj":      3,   # clausal subject
    "nsubj:pass": 4,   # passive nominal subject (checked before base split)
    "obl":        5,   # oblique nominal
    "vocative":   6,   # vocative
    "expl":       7,   # expletive
    "dislocated": 8,   # dislocated element
    "other":      9,   # everything else
}


# ══════════════════════════════════════════════════════════════════════════════
# T1 — BLiMP grammatical acceptability
# ══════════════════════════════════════════════════════════════════════════════

def load_blimp(paradigm, n=None, seed=42):
    """
    Load one BLiMP paradigm from nyu-mll/blimp (HF cache, offline-safe).
    Returns (sentences, labels): label=1 grammatical, 0 ungrammatical.
    Each paradigm: 1000 minimal pairs → 2000 sentences total before subsampling.

    n: if provided, randomly subsample to n sentences (for smoke-tests only;
       paper uses all 2000 per paradigm).
    """
    ds = load_dataset("nyu-mll/blimp", paradigm, split="train")
    sents, labels = [], []
    for row in ds:
        sents += [row["sentence_good"], row["sentence_bad"]]
        labels += [1, 0]
    if n is not None and n < len(sents):
        rng    = np.random.RandomState(seed)
        idx    = rng.choice(len(sents), n, replace=False)
        sents  = [sents[i]  for i in idx]
        labels = [labels[i] for i in idx]
    return sents, np.array(labels)


# ══════════════════════════════════════════════════════════════════════════════
# T2 — CoNLL-2003 NER (binary: entity-present=1, absent=0)
# ══════════════════════════════════════════════════════════════════════════════

def _load_conll_parquet(split="test"):
    """
    Load CoNLL-2003 from local parquet downloaded by download_data.sh.
    CoNLL-2003 NER integer tag scheme (HF encoding):
      0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
    Binary label: 1 if any token has a non-O (nonzero) tag, 0 otherwise.
    """
    path = os.path.join(_CONLL_DIR, f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CoNLL-2003 parquet not found: {path}\n"
            f"  Fix: bash code/download_data.sh"
        )
    df = pd.read_parquet(path)
    sents, labels = [], []
    for _, row in df.iterrows():
        text    = " ".join(str(t) for t in row["tokens"])
        has_ent = int(any(t != 0 for t in row["ner_tags"]))
        sents.append(text)
        labels.append(has_ent)
    return sents, labels


def load_conll(n=None, seed=42):
    """
    Paper spec: CoNLL-2003 test split, binary NER, N=3,684, natural distribution.
    No class balancing applied — paper does not specify any.

    n: if provided, randomly subsample to n sentences (for smoke-tests only;
       paper uses full test split N=3,684).
    """
    try:
        sents, labels = _load_conll_parquet(split="test")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Falling back to wikiann/en. "
              "Results will NOT match paper. Run download_data.sh to fix.")
        # wikiann is all-positive (every sentence has entities), so this
        # fallback produces degenerate binary labels — use only if desperate.
        ds = load_dataset("wikiann", "en", split="test")
        sents, labels = [], []
        for row in ds:
            text    = " ".join(str(t) for t in row["tokens"])
            has_ent = int(any(t != 0 for t in row["ner_tags"]))
            sents.append(text)
            labels.append(has_ent)

    if n is not None and n < len(sents):
        rng    = np.random.RandomState(seed)
        idx    = rng.choice(len(sents), n, replace=False)
        sents  = [sents[i]  for i in idx]
        labels = [labels[i] for i in idx]

    return sents, np.array(labels)


# ══════════════════════════════════════════════════════════════════════════════
# T3 — Universal Dependencies EN-EWT v2.13 (10-way dep classification)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_conllu(path):
    """
    Parse a CoNLL-U file. Handles both LF and CRLF line endings.
    CoNLL-U columns (tab-separated):
      0:ID  1:FORM  2:LEMMA  3:UPOS  4:XPOS  5:FEATS  6:HEAD  7:DEPREL  8:DEPS  9:MISC
    Returns: list of sentences, each = list of (token_str, deprel_str).
    Skips: comment lines (#), multi-word tokens (ID like "1-2"), empty nodes ("1.1").

    BUG FIX: uses line.rstrip() (not rstrip("\\n")) so CRLF files parse correctly.
    With rstrip("\\n"), a CRLF line becomes "token\\t...\\r" and line == "" is
    never True for blank lines, collapsing all tokens into one giant sentence.
    """
    sentences = []
    current   = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()      # strips \n, \r\n, \r, trailing spaces
            if line.startswith("#"):
                continue
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            token_id = parts[0]
            # Multi-word token: "1-2"; empty node: "1.1" — skip both
            if "-" in token_id or "." in token_id:
                continue
            token  = parts[1]
            deprel = parts[7]
            current.append((token, deprel))
    if current:     # handle missing trailing blank line
        sentences.append(current)
    return sentences


def _deprel_to_class(deprel):
    """
    Map a UD DEPREL string to one of our 10 classes.
    Tries exact match first (catches nsubj:pass → class 4),
    then the base relation before ":" (catches nsubj:outer → class 0).
    Falls back to 9 ("other").
    """
    if deprel in UD_DEP_MAP:
        return UD_DEP_MAP[deprel]
    base = deprel.split(":")[0]
    return UD_DEP_MAP.get(base, 9)


def load_ud(n=None, seed=42):
    """
    Paper spec: UD EN-EWT v2.13 train split, 10-way dep classification, N=12,543.
    Label per sentence = class of the first core-argument dep relation found
    (nsubj, obj, iobj, csubj, nsubj:pass, obl, vocative, expl, dislocated).
    Sentences where no core argument is found are labelled 9 ("other").

    No fallback — UD data must be present (run download_data.sh).

    n: if provided, randomly subsample (for smoke-tests only;
       paper uses full train split N=12,543).
    """
    conllu_path = os.path.join(_UD_DIR, "en_ewt-ud-train.conllu")
    if not os.path.exists(conllu_path):
        raise FileNotFoundError(
            f"UD EN-EWT conllu not found: {conllu_path}\n"
            f"  Fix: bash code/download_data.sh"
        )

    parsed = _parse_conllu(conllu_path)
    sents, labels = [], []
    for token_dep_pairs in parsed:
        text = " ".join(t for t, _ in token_dep_pairs)
        # Find the first token with a core-argument dep (class < 9)
        lab = 9
        for _, deprel in token_dep_pairs:
            c = _deprel_to_class(deprel)
            if c < 9:
                lab = c
                break
        sents.append(text)
        labels.append(lab)

    if n is not None and n < len(sents):
        rng    = np.random.RandomState(seed)
        idx    = rng.choice(len(sents), n, replace=False)
        sents  = [sents[i]  for i in idx]
        labels = [labels[i] for i in idx]

    return sents, np.array(labels)


# ══════════════════════════════════════════════════════════════════════════════
# H5A — SCAN add-primitive (jump) split
# ══════════════════════════════════════════════════════════════════════════════

def load_scan():
    """
    SCAN is not available as parquet on HuggingFace Hub (datasets >= 4.x).
    Returns empty lists — H5A is silently omitted from results.
    """
    return [], []


# ══════════════════════════════════════════════════════════════════════════════
# H5B — COGS compositional generalisation
# ══════════════════════════════════════════════════════════════════════════════

def load_cogs(split="gen", n=None):
    """
    Paper spec: COGS full gen split, 21,000 examples across 21 gen. types.
    Tries known HF mirrors in order. Returns (inputs, targets) or ([], []).
    """
    for repo_id in ["Punchwe/COGS", "GWHed/cogs"]:
        try:
            ds      = load_dataset(repo_id, split=split)
            inputs  = [r["sentence"] for r in ds]
            targets = [r["target"]   for r in ds]
            if n is not None and n < len(inputs):
                inputs  = inputs[:n]
                targets = targets[:n]
            return inputs, targets
        except Exception:
            pass
    print("[WARNING] COGS load failed: no working Hub dataset found. H5B skipped.")
    return [], []


# ══════════════════════════════════════════════════════════════════════════════
# Master loader
# ══════════════════════════════════════════════════════════════════════════════

def load_all_datasets(n=None):
    """
    Load all datasets and return a dict keyed by dataset name.

    n controls subsampling per dataset:
      - n=None  → full paper-specified sizes (T1:2000, T2:3684, T3:12543)
      - n=<int> → subsample each to n for smoke-tests

    NOTE: This function does NOT apply N_PROBE_SAMPLES from config.
    Callers (run_model.py, etc.) pass their own n or None for full runs.
    """
    datasets = {}

    # ── T1: BLiMP (5 paradigms from config) ──────────────────────────────────
    for paradigm in BLIMP_PARADIGMS:
        try:
            s, l = load_blimp(paradigm, n=n)
            datasets[f"blimp_{paradigm}"] = {
                "sentences": s, "labels": l, "task": "T1", "n_classes": 2
            }
        except Exception as e:
            print(f"[WARNING] BLiMP {paradigm} failed: {e}")

    # ── T2: CoNLL-2003 NER ────────────────────────────────────────────────────
    try:
        s, l = load_conll(n=n)
        datasets["conll_ner"] = {
            "sentences": s, "labels": l, "task": "T2", "n_classes": 2
        }
    except Exception as e:
        print(f"[WARNING] CoNLL failed: {e}")

    # ── T3: UD EN-EWT ─────────────────────────────────────────────────────────
    try:
        s, l = load_ud(n=n)
        datasets["ud_ewt"] = {
            "sentences": s, "labels": l, "task": "T3", "n_classes": 10
        }
    except Exception as e:
        print(f"[WARNING] UD failed: {e}")

    # ── H5A: SCAN (gracefully skipped) ───────────────────────────────────────
    si, so = load_scan()
    if si:
        datasets["scan"] = {"inputs": si, "outputs": so, "task": "H5A"}

    # ── H5B: COGS ─────────────────────────────────────────────────────────────
    try:
        ci, ct = load_cogs(n=n)
        if ci:
            datasets["cogs"] = {"inputs": ci, "targets": ct, "task": "H5B"}
    except Exception as e:
        print(f"[WARNING] COGS failed: {e}")

    return datasets


# ── spacy lazy loader (used by signals.py) ────────────────────────────────────
import spacy as _spacy
_NLP = None
def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = _spacy.load("en_core_web_sm")
    return _NLP
