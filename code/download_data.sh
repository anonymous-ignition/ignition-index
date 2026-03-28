#!/bin/bash
# download_data.sh — fetch CoNLL-2003 and UD EN-EWT from official sources
# Run on Narval LOGIN NODE (has internet). Takes ~2 min.
# Does NOT use set -e so that partial failures are reported, not silently aborted.

DATA_DIR="/home/AUTHOR/projects/ACCOUNT/AUTHOR/ignition_index/data"
mkdir -p "$DATA_DIR/conll2003"
mkdir -p "$DATA_DIR/ud_ewt"

ERRORS=0

# ─── UD EN-EWT v2.13 from official UniversalDependencies GitHub ──────────────
echo "=== Downloading UD EN-EWT v2.13 ==="
BASE="https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.13"

for split in train dev test; do
    out="$DATA_DIR/ud_ewt/en_ewt-ud-${split}.conllu"
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "  Already exists: $out"
    else
        echo "  Fetching $split..."
        wget -q -O "$out" "${BASE}/en_ewt-ud-${split}.conllu"
        # Validate: file must exist, be non-empty, and start with a # comment
        if [ ! -s "$out" ]; then
            echo "  ERROR: $out is empty after download"
            ERRORS=$((ERRORS + 1))
        elif ! head -1 "$out" | grep -q "^#"; then
            echo "  ERROR: $out does not look like a CoNLL-U file"
            ERRORS=$((ERRORS + 1))
        else
            lines=$(wc -l < "$out")
            echo "  OK: $out ($lines lines)"
        fi
    fi
done

# ─── CoNLL-2003 via HuggingFace datasets-server (parquet, no script needed) ──
echo ""
echo "=== Downloading CoNLL-2003 ==="
python3 - << 'PYEOF'
import os, sys, io, requests
import pandas as pd
from collections import defaultdict

DATA_DIR = "/home/AUTHOR/projects/ACCOUNT/AUTHOR/ignition_index/data/conll2003"
ERRORS   = 0

# HF datasets-server returns direct parquet URLs — no dataset script required.
try:
    resp = requests.get(
        "https://datasets-server.huggingface.co/parquet?dataset=conll2003",
        timeout=30
    )
    resp.raise_for_status()
except Exception as e:
    print(f"ERROR: HF datasets-server request failed: {e}")
    sys.exit(1)

parquet_files = resp.json().get("parquet_files", [])
if not parquet_files:
    print("ERROR: No parquet files returned from HF server")
    sys.exit(1)

# Group by split — there may be multiple shards per split
by_split = defaultdict(list)
for pf in parquet_files:
    by_split[pf["split"]].append(pf["url"])

print(f"  Found splits: {sorted(by_split.keys())}")

for split, urls in sorted(by_split.items()):
    out = os.path.join(DATA_DIR, f"{split}.parquet")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        df_check = pd.read_parquet(out)
        print(f"  Already exists: {out} ({len(df_check)} rows)")
        continue

    print(f"  Downloading {split} ({len(urls)} shard(s))...")
    shards = []
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            shard = pd.read_parquet(io.BytesIO(r.content))
            shards.append(shard)
            print(f"    shard {i+1}/{len(urls)}: {len(shard)} rows")
        except Exception as e:
            print(f"    ERROR downloading shard {i+1}: {e}")
            ERRORS += 1

    if not shards:
        print(f"  ERROR: No shards downloaded for {split}")
        ERRORS += 1
        continue

    df = pd.concat(shards, ignore_index=True) if len(shards) > 1 else shards[0]

    # Validate required columns
    required = {"tokens", "ner_tags"}
    missing  = required - set(df.columns)
    if missing:
        print(f"  ERROR: {split}.parquet missing columns: {missing}")
        ERRORS += 1
        continue

    df.to_parquet(out, index=False)
    print(f"  OK: {out} ({len(df)} rows, cols={df.columns.tolist()})")

if ERRORS:
    print(f"\nCoNLL-2003 finished with {ERRORS} error(s).")
    sys.exit(1)
else:
    print("CoNLL-2003 done.")
PYEOF
CONLL_EXIT=$?
if [ $CONLL_EXIT -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

# ─── Sanity checks ────────────────────────────────────────────────────────────
echo ""
echo "=== Sanity checks ==="
python3 - << 'PYEOF'
import pandas as pd, os, sys

conll_dir = "/home/AUTHOR/projects/ACCOUNT/AUTHOR/ignition_index/data/conll2003"
ud_dir    = "/home/AUTHOR/projects/ACCOUNT/AUTHOR/ignition_index/data/ud_ewt"
ERRORS    = 0

# ── CoNLL-2003 test split: paper says N=3,684 ─────────────────────────────────
try:
    df = pd.read_parquet(os.path.join(conll_dir, "test.parquet"))
    if len(df) < 3000:
        print(f"  WARN: CoNLL test has only {len(df)} rows (expected ~3684)")
    pos = sum(1 for _, r in df.iterrows() if any(t != 0 for t in r["ner_tags"]))
    neg = len(df) - pos
    print(f"  CoNLL test:  {len(df)} rows  pos={pos} neg={neg}  "
          f"cols={df.columns.tolist()}")
except Exception as e:
    print(f"  ERROR reading CoNLL test: {e}")
    ERRORS += 1

# ── UD EN-EWT train: paper says N=12,543 ─────────────────────────────────────
try:
    conllu = os.path.join(ud_dir, "en_ewt-ud-train.conllu")
    # Count sentences by blank-line separators (works for LF and CRLF)
    n_sents = 0
    prev_blank = True
    with open(conllu, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "" and not prev_blank:
                n_sents += 1
            prev_blank = (stripped == "")
    if n_sents < 12000:
        print(f"  WARN: UD train has only {n_sents} sentences (expected ~12543)")
    else:
        print(f"  UD train:    {n_sents} sentences  OK")
except Exception as e:
    print(f"  ERROR reading UD train: {e}")
    ERRORS += 1

if ERRORS:
    print(f"\nSanity checks FAILED ({ERRORS} error(s)). Fix before running experiments.")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED.")
PYEOF

echo ""
echo "Files:"
ls -lh "$DATA_DIR/ud_ewt/" "$DATA_DIR/conll2003/"

if [ $ERRORS -ne 0 ]; then
    echo ""
    echo "WARNING: $ERRORS error(s) encountered. Check output above."
    exit 1
fi

echo ""
echo "Done. Data at: $DATA_DIR"
