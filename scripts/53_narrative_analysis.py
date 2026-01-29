#!/usr/bin/env python3
# scripts/53_narrative_analysis.py
"""
Narrative analysis bundle (post-clustering, post-labeling).

Reads:
  - <run_dir>/clustered_meta.csv
  - <reports_root>/<run_name>/cluster_label_draft.csv

Writes (under <reports_root>/<run_name>/analysis/):
  - asymmetry/by_side.csv
  - asymmetry/by_type.csv
  - mirrors/by_side.csv
  - mirrors/by_type.csv
  - timeline/timeline_by_side_<freq>.csv
  - timeline/timeline_by_type_<freq>.csv
  - timeline/spikes_by_side_<freq>.csv
  - timeline/spikes_by_type_<freq>.csv
  - analysis_run.json

Example:
  python scripts/53_narrative_analysis.py --run_dir data/features/clustering/<run_name>
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_label_map(labels_csv: str) -> Dict[int, str]:
    df = pd.read_csv(labels_csv)

    if "cluster" not in df.columns:
        raise SystemExit("labels csv must contain 'cluster' column")
    if "draft_label" not in df.columns:
        raise SystemExit("labels csv must contain 'draft_label' column")

    label_map: Dict[int, str] = {}
    for _, r in df.iterrows():
        c = int(r["cluster"])
        lbl = r.get("draft_label", "")
        if pd.isna(lbl) or not str(lbl).strip() or str(lbl).strip().lower() == "nan":
            lbl = f"cluster_{c}"
        label_map[c] = str(lbl).strip()

    return label_map


def prep_meta(run_dir: str, reports_dir: str) -> Tuple[pd.DataFrame, str]:
    meta_path = os.path.join(run_dir, "clustered_meta.csv")
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing: {meta_path}")

    labels_csv = os.path.join(reports_dir, "cluster_label_draft.csv")
    if not os.path.exists(labels_csv):
        raise SystemExit(f"missing: {labels_csv} (run labeling first)")

    meta = pd.read_csv(meta_path)
    label_map = load_label_map(labels_csv)

    if "cluster" not in meta.columns:
        raise SystemExit("clustered_meta.csv must contain 'cluster'")

    meta["cluster"] = pd.to_numeric(meta["cluster"], errors="coerce").fillna(-1).astype(int)
    meta = meta[meta["cluster"] != -1].copy()
    meta["narrative"] = meta["cluster"].map(label_map)

    if "date_utc" not in meta.columns:
        raise SystemExit("clustered_meta.csv must contain 'date_utc'")

    meta["date_utc"] = pd.to_datetime(meta["date_utc"], errors="coerce", utc=True)
    meta = meta.dropna(subset=["date_utc", "narrative"]).copy()

    return meta, labels_csv


def run_asymmetry(meta: pd.DataFrame, group_by: str, min_rows: int) -> Optional[pd.DataFrame]:
    if group_by not in meta.columns:
        return None

    counts = meta.groupby([group_by, "narrative"]).size().reset_index(name="rows")
    totals = meta.groupby(group_by).size().reset_index(name="actor_total")

    df = counts.merge(totals, on=group_by, how="left")
    df["share_pct"] = (df["rows"] / df["actor_total"] * 100)

    narrative_totals = df.groupby("narrative")["rows"].sum().reset_index(name="total_rows")
    df = df.merge(narrative_totals, on="narrative", how="left")

    df = df[df["total_rows"] >= min_rows].copy()

    avg_share = df.groupby("narrative")["share_pct"].mean().reset_index(name="avg_share")
    df = df.merge(avg_share, on="narrative", how="left")

    df["dominance_score"] = (
        (df["share_pct"] / df["avg_share"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .round(2)
    )

    df["share_pct"] = df["share_pct"].round(3)
    df["avg_share"] = df["avg_share"].round(3)

    df = df.sort_values(["narrative", "dominance_score"], ascending=[True, False])
    return df


def run_mirrors(
    meta: pd.DataFrame,
    group_by: str,
    min_rows: int,
    top_narratives: int,
    min_sim: float,
) -> Optional[pd.DataFrame]:
    if group_by not in meta.columns:
        return None

    nt = meta.groupby("narrative").size().reset_index(name="total_rows")
    keep = (
        nt[nt["total_rows"] >= min_rows]
        .sort_values("total_rows", ascending=False)
        .head(top_narratives)["narrative"]
        .tolist()
    )

    if not keep:
        return None

    narratives = keep
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(narratives)
    sims = (X @ X.T).toarray()

    pairs = []
    for i in range(len(narratives)):
        for j in range(i + 1, len(narratives)):
            s = float(sims[i, j])
            if s >= min_sim:
                pairs.append((narratives[i], narratives[j], s))

    if not pairs:
        return pd.DataFrame(columns=["narrative_a", "narrative_b", "sim", "a_actor_dist", "b_actor_dist"])

    def actor_dist(narr: str) -> str:
        sub = meta[meta["narrative"] == narr]
        dist = sub[group_by].value_counts(normalize=True).head(6)
        return json.dumps({k: round(float(v), 3) for k, v in dist.items()}, ensure_ascii=False)

    rows = []
    for a, b, s in sorted(pairs, key=lambda x: x[2], reverse=True):
        rows.append(
            {
                "narrative_a": a,
                "narrative_b": b,
                "sim": round(s, 3),
                "a_actor_dist": actor_dist(a),
                "b_actor_dist": actor_dist(b),
            }
        )

    return pd.DataFrame(rows)


def run_timeline(
    meta: pd.DataFrame,
    group_by: str,
    freq: str,
    top_spikes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = meta.copy()
    df["period"] = df["date_utc"].dt.to_period(freq).dt.start_time

    gb_cols = ["period", "narrative"]
    if group_by in df.columns:
        gb_cols = ["period", group_by, "narrative"]

    ts = df.groupby(gb_cols).size().reset_index(name="rows")

    series_cols = ["narrative"]
    if group_by in df.columns:
        series_cols = [group_by, "narrative"]

    ts["mean"] = ts.groupby(series_cols)["rows"].transform("mean")
    ts["std"] = ts.groupby(series_cols)["rows"].transform("std").fillna(0)
    ts["z"] = (ts["rows"] - ts["mean"]) / ts["std"].replace(0, np.nan)
    ts["z"] = ts["z"].fillna(0).round(2)

    spikes = ts.sort_values("z", ascending=False).head(top_spikes).copy()
    ts = ts.sort_values(gb_cols)

    return ts, spikes


def main() -> None:
    ap = argparse.ArgumentParser(description="Narrative analysis bundle: asymmetry + mirrors + timeline.")
    ap.add_argument("--run_dir", required=True, help="Clustering run dir (contains clustered_meta.csv)")
    ap.add_argument("--reports_root", default="reports", help="Reports root")
    ap.add_argument("--min_rows", type=int, default=50, help="Min total rows per narrative to include")
    ap.add_argument("--top_narratives", type=int, default=120, help="Max narratives considered for mirror analysis")
    ap.add_argument("--mirror_min_sim", type=float, default=0.35, help="Min TF-IDF cosine sim between labels")
    ap.add_argument("--timeline_freq", default="W", choices=["D", "W", "M"], help="Timeline aggregation frequency")
    ap.add_argument("--top_spikes", type=int, default=60, help="Top spikes to export")
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    reports_dir = os.path.join(args.reports_root, run_name)
    ensure_dir(reports_dir)

    meta, labels_csv = prep_meta(args.run_dir, reports_dir)

    analysis_root = os.path.join(reports_dir, "analysis")
    dir_asym = os.path.join(analysis_root, "asymmetry")
    dir_mirr = os.path.join(analysis_root, "mirrors")
    dir_time = os.path.join(analysis_root, "timeline")

    ensure_dir(dir_asym)
    ensure_dir(dir_mirr)
    ensure_dir(dir_time)

    outputs: Dict[str, str] = {}

    for dim in ["side", "type"]:
        df = run_asymmetry(meta, dim, args.min_rows)
        if df is not None:
            p = os.path.join(dir_asym, f"by_{dim}.csv")
            df.to_csv(p, index=False, encoding="utf-8")
            outputs[f"asymmetry_{dim}"] = p.replace("\\", "/")

    for dim in ["side", "type"]:
        df = run_mirrors(meta, dim, args.min_rows, args.top_narratives, args.mirror_min_sim)
        if df is not None:
            p = os.path.join(dir_mirr, f"by_{dim}.csv")
            df.to_csv(p, index=False, encoding="utf-8")
            outputs[f"mirrors_{dim}"] = p.replace("\\", "/")

    for dim in ["side", "type"]:
        ts, spikes = run_timeline(meta, dim, args.timeline_freq, args.top_spikes)

        p_ts = os.path.join(dir_time, f"timeline_by_{dim}_{args.timeline_freq}.csv")
        p_sp = os.path.join(dir_time, f"spikes_by_{dim}_{args.timeline_freq}.csv")

        ts.to_csv(p_ts, index=False, encoding="utf-8")
        spikes.to_csv(p_sp, index=False, encoding="utf-8")

        outputs[f"timeline_{dim}"] = p_ts.replace("\\", "/")
        outputs[f"spikes_{dim}"] = p_sp.replace("\\", "/")

    manifest = {
        "timestamp_utc": utc_now_iso(),
        "run_dir": args.run_dir.replace("\\", "/"),
        "run_name": run_name,
        "reports_dir": reports_dir.replace("\\", "/"),
        "analysis_dir": analysis_root.replace("\\", "/"),
        "inputs": {
            "clustered_meta": os.path.join(args.run_dir, "clustered_meta.csv").replace("\\", "/"),
            "cluster_label_draft": labels_csv.replace("\\", "/"),
        },
        "params": {
            "min_rows": args.min_rows,
            "top_narratives": args.top_narratives,
            "mirror_min_sim": args.mirror_min_sim,
            "timeline_freq": args.timeline_freq,
            "top_spikes": args.top_spikes,
        },
        "outputs": outputs,
        "notes": "Mirror uses TF-IDF similarity over narrative label strings.",
    }

    man_path = os.path.join(analysis_root, "analysis_run.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Done")
    print(f"Analysis folder: {analysis_root}")
    print(f"Manifest       : {man_path}")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
