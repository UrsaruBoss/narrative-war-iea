#!/usr/bin/env python3
# scripts/30_build_embeddings.py
"""
Build multilingual sentence embeddings for the Narrative War dataset.

Reads:
  - data/processed/processed_all.csv  (must include text_clean)

Writes (under --out_dir, default data/features):
  - embeddings.npy
  - embeddings_meta.csv
  - embeddings_run.json

Example:
  python scripts/30_build_embeddings.py --input_csv data/processed/processed_all.csv --out_dir data/features
"""

import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build multilingual embeddings for Narrative War dataset.")
    ap.add_argument("--input_csv", default="data/processed/processed_all.csv", help="Path to processed_all.csv")
    ap.add_argument("--out_dir", default="data/features", help="Output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name")
    ap.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--min_chars", type=int, default=30, help="Drop texts shorter than this")
    ap.add_argument("--max_rows", type=int, default=0, help="Debug: limit rows (0 = all)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.input_csv)
    if "text_clean" not in df.columns:
        raise SystemExit("Missing column: text_clean. Run preprocess first.")

    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df["text_len_clean"] = df["text_clean"].str.len()

    before = len(df)
    df = df[df["text_len_clean"] >= args.min_chars].copy()
    dropped_short = before - len(df)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    df = df.reset_index(drop=True)
    print(f"Loaded: {before} rows | kept: {len(df)} | dropped_short(<{args.min_chars}): {dropped_short}")

    # Lazy import to keep earlier validation output visible even if model import fails.
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    texts = df["text_clean"].tolist()
    print(f"Encoding {len(texts)} texts (batch_size={args.batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    emb_path = os.path.join(args.out_dir, "embeddings.npy")
    np.save(emb_path, embeddings)

    keep_cols = [
        "conflict", "side", "type", "label", "channel",
        "message_id", "date_utc",
        "views", "forwards", "replies",
        "lang", "text_len_clean", "text_clean",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    meta = df[existing].copy()

    meta_path = os.path.join(args.out_dir, "embeddings_meta.csv")
    meta.to_csv(meta_path, index=False, encoding="utf-8")

    run = {
        "timestamp_utc": utc_now_iso(),
        "input_csv": args.input_csv,
        "rows_input": int(before),
        "rows_used": int(len(df)),
        "dropped_short": int(dropped_short),
        "min_chars": int(args.min_chars),
        "model": args.model,
        "batch_size": int(args.batch_size),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "outputs": {
            "embeddings_npy": emb_path.replace("\\", "/"),
            "meta_csv": meta_path.replace("\\", "/"),
        },
    }

    run_path = os.path.join(args.out_dir, "embeddings_run.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    print("Done")
    print(f"embeddings: {embeddings.shape} -> {emb_path}")
    print(f"meta      : {len(meta)} rows -> {meta_path}")
    print(f"manifest  : {run_path}")


if __name__ == "__main__":
    main()
