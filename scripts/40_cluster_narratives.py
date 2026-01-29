#!/usr/bin/env python3
# scripts/40_cluster_narratives.py
"""
Cluster narrative embeddings using HDBSCAN and save a versioned run folder.

Reads:
  - data/features/embeddings.npy
  - data/features/embeddings_meta.csv

Writes (under --out_root/<run_name>):
  - cluster_labels.npy
  - clustered_meta.csv
  - cluster_summary.csv
  - cluster_examples.json
  - clustering_run.json

Example:
  python scripts/40_cluster_narratives.py --min_cluster_size 35 --min_samples 10 --run_tag v1
"""

import os
import json
import argparse
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def cosine_sim_matrix_centroid(emb: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    c = centroid / (np.linalg.norm(centroid) + 1e-12)
    return emb @ c


def pick_representatives(emb: np.ndarray, idxs: np.ndarray, k: int = 7) -> np.ndarray:
    sub = emb[idxs]
    centroid = sub.mean(axis=0)
    sims = cosine_sim_matrix_centroid(sub, centroid)
    order = np.argsort(-sims)
    return idxs[order[: min(k, len(order))]]


def build_run_name(args) -> str:
    method = "hdbscan"
    sel = args.cluster_selection_method
    name = f"{method}_mcs{args.min_cluster_size}_ms{args.min_samples}_{sel}"
    if args.metric_tag:
        name += f"_{args.metric_tag}"
    if args.run_tag:
        name += f"__{args.run_tag}"
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description="Narrative clustering with HDBSCAN on multilingual embeddings.")
    ap.add_argument("--embeddings", default="data/features/embeddings.npy", help="Path to embeddings.npy")
    ap.add_argument("--meta_csv", default="data/features/embeddings_meta.csv", help="Path to embeddings_meta.csv")

    ap.add_argument("--out_root", default="data/features/clustering", help="Root directory for clustering runs")
    ap.add_argument("--run_tag", default="", help="Optional tag appended to auto run name")
    ap.add_argument("--run_name", default="", help="Override: explicit run folder name")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run folder")

    ap.add_argument("--min_cluster_size", type=int, default=35, help="HDBSCAN min_cluster_size")
    ap.add_argument("--min_samples", type=int, default=10, help="HDBSCAN min_samples")
    ap.add_argument("--cluster_selection_method", default="eom", choices=["eom", "leaf"])
    ap.add_argument("--allow_single_cluster", action="store_true")

    ap.add_argument("--max_examples_per_cluster", type=int, default=7)
    ap.add_argument("--top_clusters", type=int, default=80)

    ap.add_argument("--metric_tag", default="eucl_norm", help="Tag included in run name")
    args = ap.parse_args()

    run_name = args.run_name.strip() or build_run_name(args)
    run_dir = os.path.join(args.out_root, run_name)

    if os.path.exists(run_dir) and not args.overwrite:
        raise SystemExit(
            f"Run folder exists: {run_dir}\n"
            f"Use --overwrite or choose a different --run_tag / --run_name."
        )

    ensure_dir(run_dir)
    print(f"Run dir: {run_dir}")

    print("Loading embeddings + meta...")
    emb = np.load(args.embeddings)
    meta = pd.read_csv(args.meta_csv)

    if len(meta) != emb.shape[0]:
        raise SystemExit(f"Row mismatch: meta={len(meta)} vs embeddings={emb.shape[0]}")

    print("Running HDBSCAN...")
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=args.cluster_selection_method,
        allow_single_cluster=args.allow_single_cluster,
        prediction_data=False,
    )

    labels = clusterer.fit_predict(emb)
    meta["cluster"] = labels.astype(int)

    n_total = len(meta)
    n_noise = int((labels == -1).sum())
    clusters = sorted([c for c in np.unique(labels) if c != -1])
    n_clusters = len(clusters)

    noise_pct = (n_noise / n_total) * 100 if n_total else 0.0
    print(f"clustered: total={n_total}, clusters={n_clusters}, noise={n_noise} ({noise_pct:.2f}%)")

    labels_path = os.path.join(run_dir, "cluster_labels.npy")
    np.save(labels_path, labels)

    clustered_meta_path = os.path.join(run_dir, "clustered_meta.csv")
    meta.to_csv(clustered_meta_path, index=False, encoding="utf-8")

    print("Building cluster summary...")
    size_series = meta[meta["cluster"] != -1].groupby("cluster").size().sort_values(ascending=False)
    summary_rows = []

    for c, sz in size_series.items():
        sub = meta[meta["cluster"] == c]

        by_conflict = sub["conflict"].value_counts().to_dict() if "conflict" in sub.columns else {}
        by_side = sub["side"].value_counts().to_dict() if "side" in sub.columns else {}
        by_type = sub["type"].value_counts().to_dict() if "type" in sub.columns else {}
        by_lang = sub["lang"].value_counts().head(8).to_dict() if "lang" in sub.columns else {}

        def safe_mean(col: str) -> float:
            return float(sub[col].mean()) if col in sub.columns else 0.0

        summary_rows.append({
            "cluster": int(c),
            "size": int(sz),
            "conflict_dist": json.dumps(by_conflict, ensure_ascii=False),
            "side_dist": json.dumps(by_side, ensure_ascii=False),
            "type_dist": json.dumps(by_type, ensure_ascii=False),
            "lang_top": json.dumps(by_lang, ensure_ascii=False),
            "avg_views": round(safe_mean("views"), 2),
            "avg_forwards": round(safe_mean("forwards"), 2),
            "avg_replies": round(safe_mean("replies"), 2),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("size", ascending=False)
    summary_path = os.path.join(run_dir, "cluster_summary.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("Selecting representative examples...")
    top_clusters = df_summary.head(args.top_clusters)["cluster"].tolist()

    examples = {}
    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cluster_to_indices[int(c)].append(i)

    for c in tqdm(top_clusters, desc="Clusters"):
        idxs = np.array(cluster_to_indices[int(c)], dtype=int)
        reps = pick_representatives(emb, idxs, k=args.max_examples_per_cluster)

        sub = meta.iloc[reps].copy()
        keep_cols = [col for col in [
            "cluster", "date_utc", "conflict", "side", "type", "label", "channel",
            "views", "forwards", "replies", "lang", "text_len_clean", "text_clean"
        ] if col in sub.columns]

        examples[str(int(c))] = sub[keep_cols].to_dict(orient="records")

    examples_path = os.path.join(run_dir, "cluster_examples.json")
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    run = {
        "timestamp_utc": utc_now_iso(),
        "run_name": run_name,
        "run_dir": run_dir.replace("\\", "/"),
        "inputs": {
            "embeddings": args.embeddings.replace("\\", "/"),
            "meta_csv": args.meta_csv.replace("\\", "/"),
        },
        "params": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "cluster_selection_method": args.cluster_selection_method,
            "metric": "euclidean_on_normalized_embeddings",
            "max_examples_per_cluster": args.max_examples_per_cluster,
            "top_clusters": args.top_clusters,
        },
        "results": {
            "n_total": n_total,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": round(noise_pct, 3),
        },
        "outputs": {
            "cluster_labels_npy": labels_path.replace("\\", "/"),
            "clustered_meta_csv": clustered_meta_path.replace("\\", "/"),
            "cluster_summary_csv": summary_path.replace("\\", "/"),
            "cluster_examples_json": examples_path.replace("\\", "/"),
        },
    }

    run_path = os.path.join(run_dir, "clustering_run.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    print("Done")
    print(f"run_dir  : {run_dir}")
    print(f"labels   : {labels_path}")
    print(f"meta     : {clustered_meta_path}")
    print(f"summary  : {summary_path}")
    print(f"examples : {examples_path}")
    print(f"manifest : {run_path}")


if __name__ == "__main__":
    main()
