"""
signals.py — S1 masking, S2 embedding noise, S3 POS-constrained corruption.
Paper Sec 4.2.
"""
import numpy as np
import torch
from collections import defaultdict
from src.config import CONTENT_POS
from src.datasets import get_nlp

_POS_POOLS = {}

def apply_s1_masking(input_ids, s, pad_id, rng):
    if s >= 1.0: return input_ids.clone()
    mask = torch.tensor(rng.random(input_ids.shape) < (1.0 - s),
                        dtype=torch.bool, device=input_ids.device)
    out = input_ids.clone()
    out[mask] = pad_id
    return out

class EmbNoiseHook:
    def __init__(self, model, sigma, is_tl=True):
        self.model = model; self.sigma = sigma
        self.is_tl = is_tl; self._h = None
    def _hook(self, module, inp, out):
        return out + torch.randn_like(out) * self.sigma
    def __enter__(self):
        if self.sigma == 0.0: return self
        try:
            m = self.model.embed if self.is_tl else self.model.model.embed_tokens
        except AttributeError:
            try: m = self.model.transformer.wte
            except AttributeError: return self
        self._h = m.register_forward_hook(self._hook)
        return self
    def __exit__(self, *a):
        if self._h: self._h.remove(); self._h = None

_emb_sigma_cache = {}
def embedding_sigma(model, s, is_tl=True):
    if s >= 1.0: return 0.0
    key = id(model)
    if key not in _emb_sigma_cache:
        with torch.no_grad():
            try:   emb = model.embed.W_E if is_tl else model.model.embed_tokens.weight
            except AttributeError: emb = model.transformer.wte.weight
        _emb_sigma_cache[key] = emb.std().item()
    return (1.0 - s) * _emb_sigma_cache[key]

def _build_pos_pools(tokenizer, n_sample=3000):
    key = tokenizer.name_or_path
    if key in _POS_POOLS: return _POS_POOLS[key]
    nlp   = get_nlp()
    vocab = tokenizer.get_vocab()
    rng   = np.random.RandomState(0)
    sample_ids = rng.choice(list(vocab.values()), min(n_sample, len(vocab)), replace=False)
    pools = defaultdict(list)
    for tid in sample_ids:
        word = tokenizer.decode([int(tid)]).strip()
        if not word or len(word) > 20 or not word.isalpha(): continue
        doc = nlp(word)
        if not doc: continue
        pos = doc[0].pos_
        if pos in CONTENT_POS: pools[pos].append(word)
    for pos in CONTENT_POS:
        if len(pools[pos]) < 50:
            all_words = [w for ws in pools.values() for w in ws]
            pools[pos] = all_words if all_words else ["the", "a", "of"]
    _POS_POOLS[key] = dict(pools)
    return _POS_POOLS[key]

def apply_s3_pos_corruption(sentences, s, tokenizer, device, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    if s >= 1.0:
        enc = tokenizer(sentences, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
        return enc["input_ids"].to(device)
    nlp   = get_nlp()
    pools = _build_pos_pools(tokenizer)
    rate  = 1.0 - s
    corrupted = []
    for sent in sentences:
        doc = nlp(sent)
        new_tokens = []
        for tok in doc:
            if tok.pos_ in CONTENT_POS and rng.random() < rate:
                pool = pools.get(tok.pos_, [tok.text])
                new_tokens.append(pool[rng.randint(len(pool))])
            else:
                new_tokens.append(tok.text)
        corrupted.append(" ".join(new_tokens))
    enc = tokenizer(corrupted, return_tensors="pt", padding=True,
                    truncation=True, max_length=128)
    return enc["input_ids"].to(device)
