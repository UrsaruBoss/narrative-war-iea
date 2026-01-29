#!/usr/bin/env python3
# scripts/52_actor_narrative_profile.py
"""
Build narrative profiles per actor grouping using cluster labels.

Reads:
  - <run_dir>/clustered_meta.csv
  - reports/<run_name>/cluster_label_draft.csv (default) or --labels_csv

Writes (under reports_root/<run_name>/):
  - actor_narrative_profile.csv
  - actor_top_narratives.csv
  - channel_top_narratives.csv (if label exists)

Example:
  python scripts/52_actor_narrative_profile.py --run_dir data/features/clustering/<run_name>
"""

import os
import argparse
import pandas as pd


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_label_map(label_csv_path: str) -> dict:
    df = pd.read_csv(label_csv_path)
    if "cluster" not in df.columns:
        raise SystemExit("label CSV must contain 'cluster' column")

    label_map = {}
    for _, r in df.iterrows():
        c = int(r["cluster"])
        lbl = str(r.get("draft_label", "") or "").strip()
        if not lbl:
            lbl = f"cluster_{c}"
        label_map[c] = lbl
    return label_map


def main() -> None:
    ap = argparse.ArgumentParser(description="Build narrative profile per actor using cluster labels.")
    ap.add_argument("--run_dir", required=True, help="Clustering run folder (contains clustered_meta.csv)")
    ap.add_argument("--reports_root", default="reports", help="Reports root (versioned)")
    ap.add_argument(
        "--labels_csv",
        default="",
        help="Path to cluster_label_draft.csv (default: reports/<run_name>/cluster_label_draft.csv)",
    )
    ap.add_argument("--top_n", type=int, default=15, help="Top N narratives per grouping")
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = os.path.join(args.reports_root, run_name)
    ensure_dir(out_dir)

    meta_path = os.path.join(args.run_dir, "clustered_meta.csv")
    meta = pd.read_csv(meta_path)

    labels_csv = args.labels_csv.strip() or os.path.join(out_dir, "cluster_label_draft.csv")
    if not os.path.exists(labels_csv):
        raise SystemExit(f"labels CSV not found: {labels_csv}")

    label_map = load_label_map(labels_csv)

    meta["cluster"] = pd.to_numeric(meta["cluster"], errors="coerce").fillna(-1).astype(int)
    meta = meta[meta["cluster"] != -1].copy()

    meta["narrative"] = meta["cluster"].map(label_map).fillna(meta["cluster"].apply(lambda x: f"cluster_{x}"))

    base_cols = [c for c in ["conflict", "side", "type"] if c in meta.columns]
    if not base_cols:
        base_cols = ["narrative"]

    overall = (
        meta.groupby(base_cols + ["narrative"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
    )

    overall_path = os.path.join(out_dir, "actor_narrative_profile.csv")
    overall.to_csv(overall_path, index=False, encoding="utf-8")

    top_rows = []
    if base_cols:
        totals = meta.groupby(base_cols).size().reset_index(name="total_rows")
        merged = overall.merge(totals, on=base_cols, how="left")
        merged["share_pct"] = (merged["rows"] / merged["total_rows"] * 100).round(2)

        for _key, sub in merged.groupby(base_cols):
            sub_top = sub.sort_values("rows", ascending=False).head(args.top_n)
            top_rows.append(sub_top)

        top_df = pd.concat(top_rows, ignore_index=True) if top_rows else merged.head(args.top_n)
    else:
        top_df = overall.head(args.top_n)

    top_path = os.path.join(out_dir, "actor_top_narratives.csv")
    top_df.to_csv(top_path, index=False, encoding="utf-8")

    ch_path = None
    if "label" in meta.columns:
        ch = (
            meta.groupby(["label", "narrative"], dropna=False)
                .size()
                .reset_index(name="rows")
                .sort_values(["label", "rows"], ascending=[True, False])
        )
        ch = ch.groupby("label").head(args.top_n)
        ch_path = os.path.join(out_dir, "channel_top_narratives.csv")
        ch.to_csv(ch_path, index=False, encoding="utf-8")

    print("Done")
    print(f"Run dir  : {args.run_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Overall  : {overall_path}")
    print(f"Top N    : {top_path}")
    if ch_path:
        print(f"Channels : {ch_path}")


if __name__ == "__main__":
    main()
