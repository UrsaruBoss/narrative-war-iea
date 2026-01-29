#!/usr/bin/env python3
# scripts/50_analyze_clusters.py
"""
Analyze a clustering run (noise vs clusters, coverage, top clusters by actor).

Reads (from --run_dir):
  - clustered_meta.csv
  - cluster_summary.csv
  - clustering_run.json (optional)

Writes (under --reports_root/<run_name>):
  - cluster_coverage.csv
  - noise_breakdown.csv
  - top_clusters_by_actor.csv (if label exists)
  - report_summary.json

Example:
  python scripts/50_analyze_clusters.py --run_dir data/features/clustering/<run_name>
"""

import os
import json
import argparse
from datetime import datetime, timezone

import pandas as pd


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_run(run_dir: str):
    meta_path = os.path.join(run_dir, "clustered_meta.csv")
    summary_path = os.path.join(run_dir, "cluster_summary.csv")
    run_path = os.path.join(run_dir, "clustering_run.json")

    meta = pd.read_csv(meta_path)
    summary = pd.read_csv(summary_path)

    run = {}
    if os.path.exists(run_path):
        with open(run_path, "r", encoding="utf-8") as f:
            run = json.load(f)

    return meta, summary, run


def pct(x: int, total: int) -> float:
    return round((x / total) * 100, 2) if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze clustering run and write report CSV/JSON artifacts.")
    ap.add_argument("--run_dir", required=True, help="Clustering run folder (contains clustered_meta.csv etc.)")
    ap.add_argument("--reports_root", default="reports", help="Root directory for analysis reports (versioned per run)")
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = os.path.join(args.reports_root, run_name)
    ensure_dir(out_dir)

    meta, _summary, run = load_run(args.run_dir)

    total = len(meta)
    noise = int((meta["cluster"] == -1).sum())
    clustered = total - noise

    print("CLUSTER RUN ANALYSIS")
    print("=" * 70)
    print(f"Run dir        : {args.run_dir}")
    if run:
        print(f"Run name       : {run.get('run_name', '')}")
        print(f"Params         : {run.get('params', {})}")
    print(f"Total rows     : {total}")
    print(f"Clustered rows : {clustered} ({pct(clustered, total)}%)")
    print(f"Noise rows     : {noise} ({pct(noise, total)}%)")

    # Coverage by (conflict, side, type, label)
    group_cols = [c for c in ["conflict", "side", "type", "label"] if c in meta.columns]
    cov = (
        meta.assign(is_noise=(meta["cluster"] == -1))
            .groupby(group_cols, dropna=False)
            .agg(rows=("cluster", "size"), noise_rows=("is_noise", "sum"))
            .reset_index()
    )
    cov["clustered_rows"] = cov["rows"] - cov["noise_rows"]
    cov["noise_pct"] = cov.apply(lambda r: pct(int(r["noise_rows"]), int(r["rows"])), axis=1)
    cov["clustered_pct"] = cov.apply(lambda r: pct(int(r["clustered_rows"]), int(r["rows"])), axis=1)

    cov_path = os.path.join(out_dir, "cluster_coverage.csv")
    cov.sort_values(["noise_pct", "rows"], ascending=[False, False]).to_csv(cov_path, index=False, encoding="utf-8")

    # Noise breakdown by type/side/conflict (where available)
    breakdown_cols = [c for c in ["type", "side", "conflict"] if c in meta.columns]
    noise_bd = (
        meta.assign(is_noise=(meta["cluster"] == -1))
            .groupby(breakdown_cols, dropna=False)
            .agg(rows=("cluster", "size"), noise_rows=("is_noise", "sum"))
            .reset_index()
    )
    noise_bd["noise_pct"] = noise_bd.apply(lambda r: pct(int(r["noise_rows"]), int(r["rows"])), axis=1)

    noise_path = os.path.join(out_dir, "noise_breakdown.csv")
    noise_bd.sort_values(["noise_pct", "rows"], ascending=[False, False]).to_csv(noise_path, index=False, encoding="utf-8")

    # Top clusters by actor (label)
    top_actor_path = None
    clustered_meta = meta[meta["cluster"] != -1].copy()
    if "label" in clustered_meta.columns:
        top_actor = (
            clustered_meta.groupby(["label", "cluster"])
                          .size()
                          .reset_index(name="rows")
                          .sort_values(["label", "rows"], ascending=[True, False])
        )
        top_actor = top_actor.groupby("label").head(10)
        top_actor_path = os.path.join(out_dir, "top_clusters_by_actor.csv")
        top_actor.to_csv(top_actor_path, index=False, encoding="utf-8")

    report = {
        "run_dir": args.run_dir,
        "run_name": run_name,
        "total_rows": int(total),
        "clustered_rows": int(clustered),
        "noise_rows": int(noise),
        "noise_pct": pct(noise, total),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "cluster_coverage_csv": cov_path.replace("\\", "/"),
            "noise_breakdown_csv": noise_path.replace("\\", "/"),
            "top_clusters_by_actor_csv": (top_actor_path.replace("\\", "/") if top_actor_path else ""),
        },
    }

    report_path = os.path.join(out_dir, "report_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nReports saved")
    print(f"- {cov_path}")
    print(f"- {noise_path}")
    if top_actor_path:
        print(f"- {top_actor_path}")
    print(f"- {report_path}")

    print("\nQuick highlights (highest noise_pct per actor)")
    print(cov.sort_values("noise_pct", ascending=False).head(8).to_string(index=False))

    print("\nDone")


if __name__ == "__main__":
    main()
