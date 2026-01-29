#!/usr/bin/env python3
# scripts/10_quick_audit.py
"""
Quick audit for raw Telegram CSV exports.

Reads:
  - data/raw/**/*.csv

Writes:
  - data/audit/audit_summary.csv
  - data/audit/audit_top_messages.csv

Example:
  python scripts/10_quick_audit.py
"""

import os
import glob
import pandas as pd


RAW_DIR = "data/raw"
OUT_DIR = "data/audit"
TOP_N = 5


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csvs(root: str) -> list[str]:
    return glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)


def audit_file(path: str):
    df = pd.read_csv(path)

    if df.empty:
        return None

    # --- normalize timestamps ---
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")

    rows = len(df)
    empty_text_pct = (df["text_len"] == 0).mean() * 100

    date_min = df["date_utc"].min()
    date_max = df["date_utc"].max()

    days = max((date_max - date_min).days, 1)
    per_day = rows / days

    meta = {
        "file": path.replace("\\", "/"),
        "conflict": df["conflict"].iloc[0],
        "side": df["side"].iloc[0],
        "type": df["type"].iloc[0],
        "label": df["label"].iloc[0],
        "rows": rows,
        "date_min": date_min.isoformat() if pd.notna(date_min) else "",
        "date_max": date_max.isoformat() if pd.notna(date_max) else "",
        "days_covered": days,
        "rows_per_day": round(per_day, 2),
        "empty_text_pct": round(empty_text_pct, 2),
    }

    # --- top messages (simple triage views + forwards) ---
    top_views = df.sort_values("views", ascending=False).head(TOP_N).assign(metric="views")
    top_fwds = df.sort_values("forwards", ascending=False).head(TOP_N).assign(metric="forwards")
    top = pd.concat([top_views, top_fwds], ignore_index=True)

    return meta, top


def main() -> None:
    ensure_dir(OUT_DIR)

    files = load_csvs(RAW_DIR)
    summaries = []
    tops = []

    print("DATASET AUDIT")
    print("=" * 60)

    for f in files:
        res = audit_file(f)
        if not res:
            print(f"EMPTY: {f}")
            continue

        meta, top = res
        summaries.append(meta)
        tops.append(top)

        print(f"\n{meta['label']} ({meta['conflict']} / {meta['side']} / {meta['type']})")
        print(f"rows       : {meta['rows']}")
        print(f"period     : {meta['date_min']} -> {meta['date_max']}")
        print(f"rows/day   : {meta['rows_per_day']}")
        print(f"empty text : {meta['empty_text_pct']}%")

    # --- write outputs ---
    if summaries:
        pd.DataFrame(summaries).to_csv(os.path.join(OUT_DIR, "audit_summary.csv"), index=False)

    if tops:
        df_top = pd.concat(tops, ignore_index=True)
        df_top = df_top[
            [
                "conflict",
                "side",
                "type",
                "label",
                "date_utc",
                "views",
                "forwards",
                "replies",
                "text",
                "metric",
            ]
        ]
        df_top.to_csv(os.path.join(OUT_DIR, "audit_top_messages.csv"), index=False)

    print("\n" + "=" * 60)
    print("Audit complete")
    print(f"Saved: {OUT_DIR}/audit_summary.csv")
    print(f"Saved: {OUT_DIR}/audit_top_messages.csv")


if __name__ == "__main__":
    main()
