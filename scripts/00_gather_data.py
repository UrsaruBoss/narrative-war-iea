#!/usr/bin/env python3
# scripts/00_gather_data.py
"""
Collect Telegram posts from channels defined in a YAML config (Telethon).

Reads:
  - config/channels.yaml
  - .env (TG_API_ID, TG_API_HASH)

Writes:
  - data/raw/<conflict>/<side>/*.csv
  - optional: data/raw/<conflict>/<side>/*.jsonl
  - Telethon session file (default: narrative_session.session)

Example:
  python scripts/00_gather_data.py --config config/channels.yaml --out_dir data --days 30 --limit 5000
"""

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_date(s: str) -> datetime:
    # Accept: YYYY-MM-DD or full ISO
    if len(s) == 10:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    return s.strip("_") or "channel"


def clean_text(t: str) -> str:
    # Minimal cleaning (keep meaning)
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def msg_replies_count(m) -> int:
    try:
        return int(m.replies.replies) if m.replies else 0
    except Exception:
        return 0


def msg_has_media(m) -> bool:
    return bool(getattr(m, "media", None))


@dataclass
class ChannelSpec:
    conflict: str
    side: str
    ctype: str  # official/media
    label: str
    channel: str


def load_channels_yaml(path: str) -> List[ChannelSpec]:
    # --- load channel configuration ---
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    specs: List[ChannelSpec] = []
    for conflict, sides in cfg.items():
        for side, types in sides.items():
            for ctype, entry in types.items():
                specs.append(
                    ChannelSpec(
                        conflict=conflict,
                        side=side,
                        ctype=ctype,
                        label=str(entry.get("label") or f"{conflict}_{side}_{ctype}"),
                        channel=str(entry["channel"]),
                    )
                )
    return specs


async def collect_channel(
    client: TelegramClient,
    spec: ChannelSpec,
    out_dir: str,
    limit: int,
    since_dt: Optional[datetime],
    until_dt: Optional[datetime],
    include_non_text: bool,
    min_sleep: float,
    max_sleep: float,
    export_jsonl: bool,
) -> Tuple[str, int]:
    """
    Collect messages for one channel and export CSV (and optionally JSONL).
    Returns (output_csv_path, rows_count).
    """

    # --- output paths ---
    base_dir = os.path.join(out_dir, "raw", spec.conflict, spec.side)
    ensure_dir(base_dir)

    base_name = f"{safe_filename(spec.label)}__{safe_filename(spec.ctype)}"
    csv_path = os.path.join(base_dir, f"{base_name}.csv")
    jsonl_path = os.path.join(base_dir, f"{base_name}.jsonl")

    print(f"\nCollecting {spec.conflict}/{spec.side}/{spec.ctype} -> {spec.channel}")
    print(f"Output: {csv_path}")

    # --- resolve channel entity ---
    try:
        entity = await client.get_entity(spec.channel)
    except Exception as e:
        print(f"Failed to resolve channel {spec.channel}: {e}")
        return (csv_path, 0)

    rows: List[Dict[str, Any]] = []
    seen_ids = set()

    fetched = 0
    kept = 0

    # Telethon iterates newest -> oldest by default; we break once we pass since_dt.
    try:
        async for m in client.iter_messages(entity, limit=limit):
            print(f"\rFetched: {fetched}, Kept: {kept}", end="", flush=True)
            fetched += 1

            if not getattr(m, "date", None):
                continue

            dt = m.date
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

            # --- apply time window ---
            if until_dt and dt > until_dt:
                continue
            if since_dt and dt < since_dt:
                break

            text = getattr(m, "message", None) or ""
            has_media = msg_has_media(m)

            if not text and (not include_non_text):
                continue

            mid = int(getattr(m, "id", 0) or 0)
            if mid and mid in seen_ids:
                continue
            if mid:
                seen_ids.add(mid)

            text_clean = clean_text(text) if text else ""

            rows.append(
                {
                    "conflict": spec.conflict,
                    "side": spec.side,
                    "type": spec.ctype,
                    "label": spec.label,
                    "channel": spec.channel,
                    "message_id": mid,
                    "date_utc": dt.isoformat(),
                    "text": text_clean,
                    "text_len": len(text_clean),
                    "views": int(getattr(m, "views", 0) or 0),
                    "forwards": int(getattr(m, "forwards", 0) or 0),
                    "replies": msg_replies_count(m),
                    "has_media": bool(has_media),
                    "edit_date_utc": (
                        m.edit_date.astimezone(timezone.utc).isoformat()
                        if getattr(m, "edit_date", None)
                        else ""
                    ),
                }
            )
            kept += 1

            # --- soft throttling (rate-limit safety) ---
            if min_sleep or max_sleep:
                s = min_sleep if fetched % 3 else max_sleep
                if s > 0:
                    time.sleep(s)

    except FloodWaitError as e:
        print(f"\nFloodWaitError: need to wait {e.seconds}s. Sleeping then continuing (best effort).")
        time.sleep(int(e.seconds) + 1)
    except RPCError as e:
        print(f"\nTelegram RPC error on {spec.channel}: {e}")
    except Exception as e:
        print(f"\nUnexpected error on {spec.channel}: {e}")

    # --- write outputs ---
    if not rows:
        print("\nNo rows collected (check channel access / filters).")
        pd.DataFrame([]).to_csv(csv_path, index=False, encoding="utf-8")
        if export_jsonl:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                pass
        return (csv_path, 0)

    df = pd.DataFrame(rows)

    # Sort ascending by time for downstream processing
    df = df.sort_values(by=["date_utc", "message_id"], ascending=[True, True])

    df.to_csv(csv_path, index=False, encoding="utf-8")

    if export_jsonl:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in df.to_dict(orient="records"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Fetched≈{fetched}, Kept={kept}, Saved={len(df)} rows")
    return (csv_path, len(df))


async def main_async(args) -> int:
    # --- load credentials ---
    load_dotenv()
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")

    if not api_id or not api_hash:
        print("Missing TG_API_ID / TG_API_HASH in environment (.env).")
        return 2

    api_id_int = int(api_id)

    # --- select channels from config + filters ---
    specs = load_channels_yaml(args.config)
    if args.only_conflict:
        specs = [s for s in specs if s.conflict in args.only_conflict]
    if args.only_side:
        specs = [s for s in specs if s.side in args.only_side]
    if args.only_type:
        specs = [s for s in specs if s.ctype in args.only_type]

    if not specs:
        print("No channels selected after filters.")
        return 3

    # --- time window ---
    since_dt = None
    until_dt = None

    if args.days and args.days > 0:
        since_dt = utc_now() - timedelta(days=args.days)

    if args.since:
        since_dt = parse_iso_date(args.since)

    if args.until:
        until_dt = parse_iso_date(args.until)

    if since_dt and until_dt and since_dt > until_dt:
        print("Invalid window: since > until.")
        return 4

    ensure_dir(args.out_dir)

    client = TelegramClient(args.session_name, api_id_int, api_hash)

    # Collect sequentially (safer for rate limits)
    total = 0
    async with client:
        for spec in specs:
            _, n = await collect_channel(
                client=client,
                spec=spec,
                out_dir=args.out_dir,
                limit=args.limit,
                since_dt=since_dt,
                until_dt=until_dt,
                include_non_text=args.include_non_text,
                min_sleep=args.min_sleep,
                max_sleep=args.max_sleep,
                export_jsonl=args.jsonl,
            )
            total += n

    print(f"\nAll done. Total rows saved: {total}")
    print(f"Output root: {args.out_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Telegram channel collector (Narrative War Lab) - Telethon based, config-driven."
    )

    p.add_argument("--config", type=str, default="config/channels.yaml", help="YAML config with channels.")
    p.add_argument("--out_dir", type=str, default="data", help="Output directory root (writes data/raw/...).")
    p.add_argument("--session_name", type=str, default="narrative_session", help="Telethon session name (file).")

    p.add_argument("--limit", type=int, default=5000, help="Max messages per channel (newest->oldest).")
    p.add_argument("--days", type=int, default=0, help="Collect only last N days (UTC). 0 disables.")
    p.add_argument("--since", type=str, default="", help="Collect messages with date >= since (YYYY-MM-DD or ISO).")
    p.add_argument("--until", type=str, default="", help="Collect messages with date <= until (YYYY-MM-DD or ISO).")

    p.add_argument("--include_non_text", action="store_true", help="Include non-text posts (media-only).")
    p.add_argument("--jsonl", action="store_true", help="Also export JSONL next to CSV.")

    p.add_argument("--only_conflict", nargs="*", default=[], help="Filter conflicts (exact match).")
    p.add_argument("--only_side", nargs="*", default=[], help="Filter sides (exact match).")
    p.add_argument("--only_type", nargs="*", default=[], help="Filter types: official media")

    p.add_argument("--min_sleep", type=float, default=0.0, help="Sleep seconds between some messages.")
    p.add_argument("--max_sleep", type=float, default=0.2, help="Occasional longer sleep seconds.")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    args.since = args.since.strip() if args.since else ""
    args.until = args.until.strip() if args.until else ""
    args.only_conflict = [x.strip() for x in (args.only_conflict or []) if x.strip()]
    args.only_side = [x.strip() for x in (args.only_side or []) if x.strip()]
    args.only_type = [x.strip() for x in (args.only_type or []) if x.strip()]

    import asyncio
    code = asyncio.run(main_async(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
