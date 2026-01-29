#!/usr/bin/env python3
# scripts/90_summary.py
"""Create a compact run summary (JSON + Markdown).

Reads:
  - <run_dir>/clustered_meta.csv
  - <run_dir>/clustering_run.json (optional)
  - <reports_root>/<run_name>/cluster_label_draft.csv (optional)
  - <reports_root>/<run_name>/narrative_analysis/* (optional)

Writes:
  - <reports_root>/<run_name>/<out_name>.json
  - <reports_root>/<run_name>/<out_name>.md
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_json_safe(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return ""


def fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return ""


def infer_run_name(run_dir: str) -> str:
    return os.path.basename(os.path.normpath(run_dir))


def summarize_clustered_meta(df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}

    out["rows_total"] = int(len(df))

    if "cluster" in df.columns:
        c = pd.to_numeric(df["cluster"], errors="coerce").fillna(-999).astype(int)
        out["rows_noise"] = int((c == -1).sum())
        out["noise_pct"] = round((out["rows_noise"] / max(1, out["rows_total"])) * 100, 3)
        out["clusters_n"] = int(df.loc[c != -1, "cluster"].nunique())
    else:
        out["rows_noise"] = 0
        out["noise_pct"] = 0.0
        out["clusters_n"] = 0

    if "date_utc" in df.columns:
        d = pd.to_datetime(df["date_utc"], errors="coerce", utc=True).dropna()
        if not d.empty:
            out["period_start"] = d.min().isoformat()
            out["period_end"] = d.max().isoformat()

    for col in ["conflict", "side", "type", "label", "lang"]:
        if col in df.columns:
            out[f"{col}_top"] = df[col].astype(str).value_counts().head(12).to_dict()

    for col in ["views", "forwards", "replies"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = float(s.mean()) if s.notna().any() else 0.0
            out[f"{col}_median"] = float(s.median()) if s.notna().any() else 0.0

    return out


def summarize_labels(df_labels: Optional[pd.DataFrame]) -> dict:
    out: dict[str, Any] = {}
    if df_labels is None or df_labels.empty:
        return out

    out["clusters_in_draft"] = int(len(df_labels))

    if "draft_label" in df_labels.columns:
        filled = df_labels["draft_label"].astype(str).fillna("").str.strip()
        good = (~filled.eq("")) & (~filled.str.lower().eq("nan"))
        out["draft_label_filled"] = int(good.sum())
        out["draft_label_fill_pct"] = round((out["draft_label_filled"] / max(1, out["clusters_in_draft"])) * 100, 2)

    return out


def summarize_mirrors(df: Optional[pd.DataFrame], top_k: int = 25) -> dict:
    if df is None or df.empty:
        return {}
    keep = [c for c in ["narrative_a", "narrative_b", "sim", "a_actor_dist", "b_actor_dist"] if c in df.columns]
    top = df.sort_values("sim", ascending=False).head(top_k)[keep]
    return {"top_pairs": top.to_dict(orient="records")}


def summarize_spikes(df: Optional[pd.DataFrame], top_k: int = 30) -> dict:
    if df is None or df.empty:
        return {}
    keep = [c for c in ["period", "side", "type", "narrative", "rows", "mean", "std", "z"] if c in df.columns]
    top = df.sort_values("z", ascending=False).head(top_k)[keep]
    return {"top_spikes": top.to_dict(orient="records")}


def summarize_asymmetry(df: Optional[pd.DataFrame], group_col: str, top_k: int = 12) -> dict:
    if df is None or df.empty:
        return {}

    out: dict[str, Any] = {}

    if group_col in df.columns and "dominance_score" in df.columns:
        top = (
            df.sort_values(["dominance_score", "rows"], ascending=[False, False])
            .groupby(group_col)
            .head(top_k)
        )
        cols = [c for c in [group_col, "narrative", "rows", "share_pct", "dominance_score"] if c in top.columns]
        out["top_dominant_by_actor"] = top[cols].to_dict(orient="records")

    if "narrative" in df.columns and "dominance_score" in df.columns:
        g = (
            df.groupby("narrative")["dominance_score"].max()
            .reset_index(name="max_dominance")
            .sort_values("max_dominance", ascending=False)
            .head(top_k * 2)
        )
        out["most_asymmetric_narratives"] = g.to_dict(orient="records")

    return out


def to_markdown(summary: dict) -> str:
    lines: list[str] = []

    lines.append("# Narrative War Summary")
    lines.append("")
    lines.append(f"- generated_utc: `{summary.get('timestamp_utc','')}`")
    lines.append(f"- run_dir: `{summary.get('run_dir','')}`")
    lines.append(f"- run_name: `{summary.get('run_name','')}`")
    lines.append(f"- reports_dir: `{summary.get('reports_dir','')}`")
    lines.append("")

    ds = summary.get("dataset", {})
    if ds:
        lines.append("## Dataset overview")
        lines.append(f"- rows_total: **{fmt_int(ds.get('rows_total'))}**")
        lines.append(f"- clusters_n: **{fmt_int(ds.get('clusters_n'))}**")
        lines.append(f"- noise: **{fmt_int(ds.get('rows_noise'))}** ({fmt_pct(ds.get('noise_pct'))})")
        if ds.get("period_start") and ds.get("period_end"):
            lines.append(f"- period: `{ds['period_start']}` -> `{ds['period_end']}`")
        lines.append("")

    lab = summary.get("labeling", {})
    if lab:
        lines.append("## Labeling status")
        lines.append(f"- clusters_in_draft: **{fmt_int(lab.get('clusters_in_draft'))}**")
        if "draft_label_filled" in lab:
            lines.append(
                f"- draft_label_filled: **{fmt_int(lab.get('draft_label_filled'))}** ({fmt_pct(lab.get('draft_label_fill_pct'))})"
            )
        lines.append("")

    ana = summary.get("analysis", {})
    if ana:
        lines.append("## 53 Asymmetry highlights")
        for dim in ["side", "type"]:
            a = ana.get(f"53_{dim}", {})
            if not a:
                continue
            lines.append(f"### By {dim}")
            most = a.get("most_asymmetric_narratives", [])
            for r in most[:12]:
                lines.append(f"- {r.get('narrative')} (max_dominance={r.get('max_dominance')})")
            lines.append("")

        lines.append("## 54 Mirror narratives (top pairs)")
        for dim in ["side", "type"]:
            m = ana.get(f"54_{dim}", {})
            if not m:
                continue
            lines.append(f"### By {dim}")
            for r in m.get("top_pairs", [])[:12]:
                lines.append(f"- sim={r.get('sim')} | {r.get('narrative_a')} <-> {r.get('narrative_b')}")
            lines.append("")

        lines.append("## 55 Timeline spikes (top)")
        for dim in ["side", "type"]:
            s = ana.get(f"55_{dim}", {})
            if not s:
                continue
            lines.append(f"### By {dim}")
            for r in s.get("top_spikes", [])[:12]:
                who = r.get(dim, "")
                lines.append(f"- z={r.get('z')} rows={r.get('rows')} | {r.get('period')} | {who} | {r.get('narrative')}")
            lines.append("")

    lines.append("## Files")
    for k, v in (summary.get("files", {}) or {}).items():
        if v:
            lines.append(f"- {k}: `{v}`")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a run (JSON + Markdown)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--reports_root", default="reports")
    ap.add_argument("--analysis_subdir", default="narrative_analysis")
    ap.add_argument("--out_name", default="summary")
    ap.add_argument("--top_k", type=int, default=25)
    args = ap.parse_args()

    run_name = infer_run_name(args.run_dir)

    reports_dir = os.path.join(args.reports_root, run_name)
    analysis_dir = os.path.join(reports_dir, args.analysis_subdir)
    ensure_dir(reports_dir)

    clustered_meta_path = os.path.join(args.run_dir, "clustered_meta.csv")
    clustering_run_path = os.path.join(args.run_dir, "clustering_run.json")
    label_draft_path = os.path.join(reports_dir, "cluster_label_draft.csv")

    a53_side = os.path.join(analysis_dir, "53_narrative_asymmetry_side.csv")
    a53_type = os.path.join(analysis_dir, "53_narrative_asymmetry_type.csv")
    a54_side = os.path.join(analysis_dir, "54_mirror_pairs_side.csv")
    a54_type = os.path.join(analysis_dir, "54_mirror_pairs_type.csv")
    a55_spikes_side = os.path.join(analysis_dir, "55_top_spikes_side.csv")
    a55_spikes_type = os.path.join(analysis_dir, "55_top_spikes_type.csv")

    df_meta = read_csv_safe(clustered_meta_path)
    df_labels = read_csv_safe(label_draft_path)

    summary: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "run_name": run_name,
        "run_dir": args.run_dir.replace("\\", "/"),
        "reports_dir": reports_dir.replace("\\", "/"),
        "analysis_dir": analysis_dir.replace("\\", "/"),
        "dataset": {},
        "labeling": {},
        "analysis": {},
        "files": {},
    }

    if df_meta is not None:
        summary["dataset"] = summarize_clustered_meta(df_meta)

    if df_labels is not None:
        summary["labeling"] = summarize_labels(df_labels)

    clustering_run = read_json_safe(clustering_run_path)
    if clustering_run:
        summary["clustering_run"] = clustering_run

    df53s = read_csv_safe(a53_side)
    df53t = read_csv_safe(a53_type)
    if df53s is not None:
        summary["analysis"]["53_side"] = summarize_asymmetry(df53s, "side", top_k=12)
    if df53t is not None:
        summary["analysis"]["53_type"] = summarize_asymmetry(df53t, "type", top_k=12)

    df54s = read_csv_safe(a54_side)
    df54t = read_csv_safe(a54_type)
    if df54s is not None:
        summary["analysis"]["54_side"] = summarize_mirrors(df54s, top_k=min(args.top_k, 50))
    if df54t is not None:
        summary["analysis"]["54_type"] = summarize_mirrors(df54t, top_k=min(args.top_k, 50))

    df55s = read_csv_safe(a55_spikes_side)
    df55t = read_csv_safe(a55_spikes_type)
    if df55s is not None:
        summary["analysis"]["55_side"] = summarize_spikes(df55s, top_k=min(args.top_k, 60))
    if df55t is not None:
        summary["analysis"]["55_type"] = summarize_spikes(df55t, top_k=min(args.top_k, 60))

    summary["files"] = {
        "clustered_meta_csv": clustered_meta_path.replace("\\", "/") if os.path.exists(clustered_meta_path) else "",
        "clustering_run_json": clustering_run_path.replace("\\", "/") if os.path.exists(clustering_run_path) else "",
        "cluster_label_draft_csv": label_draft_path.replace("\\", "/") if os.path.exists(label_draft_path) else "",
        "53_asymmetry_side": a53_side.replace("\\", "/") if os.path.exists(a53_side) else "",
        "53_asymmetry_type": a53_type.replace("\\", "/") if os.path.exists(a53_type) else "",
        "54_mirror_pairs_side": a54_side.replace("\\", "/") if os.path.exists(a54_side) else "",
        "54_mirror_pairs_type": a54_type.replace("\\", "/") if os.path.exists(a54_type) else "",
        "55_spikes_side": a55_spikes_side.replace("\\", "/") if os.path.exists(a55_spikes_side) else "",
        "55_spikes_type": a55_spikes_type.replace("\\", "/") if os.path.exists(a55_spikes_type) else "",
    }

    out_json = os.path.join(reports_dir, f"{args.out_name}.json")
    out_md = os.path.join(reports_dir, f"{args.out_name}.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(to_markdown(summary))

    print("Done")
    print(f"JSON: {out_json}")
    print(f"MD  : {out_md}")


if __name__ == "__main__":
    main()