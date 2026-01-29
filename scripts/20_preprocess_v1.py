#!/usr/bin/env python3
# scripts/20_preprocess_v1.py
"""
Preprocess raw Telegram CSV files into cleaned, deduplicated outputs.

Steps:
  - light text cleaning
  - drop empty text after cleaning
  - global deduplication (SHA1 on cleaned text)
  - language detection (best-effort)

Reads:
  - data/raw/**/*.csv

Writes:
  - data/processed/<same tree as raw>/*.csv
  - data/processed/processed_all.csv

Example:
  python scripts/20_preprocess_v1.py
"""

import os
import glob
import hashlib
import re
from collections import defaultdict

import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm


DetectorFactory.seed = 42  # reproducibility

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clean_text_light(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def text_hash(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def detect_lang_safe(t: str) -> str:
    try:
        if len(t) < 20:
            return "unk"
        return detect(t)
    except Exception:
        return "unk"


def main() -> None:
    ensure_dir(OUT_DIR)

    files = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)
    if not files:
        raise SystemExit("No raw CSV files found under data/raw")

    print(f"Found {len(files)} raw CSV files")

    all_rows = []
    seen_hashes = set()
    stats = defaultdict(int)

    # --- process each raw file ---
    for path in tqdm(files, desc="Preprocess"):
        df = pd.read_csv(path)
        if df.empty:
            continue

        # --- clean text and drop empty ---
        df["text_clean"] = df["text"].apply(clean_text_light)

        before = len(df)
        df = df[df["text_clean"].str.len() > 0]
        stats["empty_dropped"] += before - len(df)

        # --- global dedup by cleaned text hash ---
        df["text_hash"] = df["text_clean"].apply(text_hash)

        keep_mask = []
        for h in df["text_hash"]:
            if h in seen_hashes:
                keep_mask.append(False)
                stats["duplicates_dropped"] += 1
            else:
                keep_mask.append(True)
                seen_hashes.add(h)

        df = df[keep_mask]

        # --- language detection (best-effort) ---
        df["lang"] = df["text_clean"].apply(detect_lang_safe)

        # --- write per-channel processed file (mirror raw tree) ---
        rel = path.replace(RAW_DIR, "").lstrip("\\/").replace("\\", "/")
        out_path = os.path.join(OUT_DIR, rel)
        ensure_dir(os.path.dirname(out_path))
        df.to_csv(out_path, index=False, encoding="utf-8")

        stats["rows_kept"] += len(df)
        all_rows.append(df)

    # --- merge all processed rows ---
    if not all_rows:
        raise SystemExit("No non-empty processed rows produced (check raw inputs).")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, "processed_all.csv"), index=False, encoding="utf-8")

    # --- summary ---
    print("\nPREPROCESS V1 SUMMARY")
    print("=" * 40)
    print(f"Rows kept          : {stats['rows_kept']}")
    print(f"Empty dropped      : {stats['empty_dropped']}")
    print(f"Duplicates dropped : {stats['duplicates_dropped']}")
    print(f"Languages detected : {df_all['lang'].value_counts().to_dict()}")
    print(f"Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
