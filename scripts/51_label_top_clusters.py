#!/usr/bin/env python3
# scripts/51_label_top_clusters.py
"""
Compute TF-IDF top terms per cluster to support manual labeling.
Optionally translate terms/examples using googletrans with a simple disk cache.
Optionally produce a neutral draft label using rule-based heuristics.

Reads:
  - <run_dir>/clustered_meta.csv
  - <run_dir>/cluster_summary.csv

Writes (under reports_root/<run_name>/):
  - cluster_label_draft.csv
  - cluster_top_terms.json
  - translate_cache_<lang>.json (optional)

Example:
  python scripts/51_label_top_clusters.py --run_dir data/features/clustering/<run_name> --translate --autolabel
"""

import os
import json
import argparse
import hashlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from googletrans import Translator


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except Exception:
        return {}


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class FreeGoogleTranslator:
    """
    Minimal googletrans wrapper with on-disk cache to reduce repeated requests.
    Note: googletrans can break occasionally (unofficial). Failures fall back to original text.
    """

    def __init__(self, cache_path: str, target: str = "en"):
        self.translator = Translator()
        self.target = target
        self.cache_path = cache_path
        self.cache = {}

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def translate_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        key = f"{self.target}:{sha1(text)}"
        if key in self.cache:
            return self.cache[key]

        try:
            out = self.translator.translate(text, dest=self.target).text
            if not isinstance(out, str) or not out.strip():
                out = text
        except Exception:
            out = text

        self.cache[key] = out
        return out

    def save(self) -> None:
        if not self.cache_path:
            return
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("\n", " ").split())


def _has_any(text: str, keys) -> bool:
    return any(k in text for k in keys)


def autolabel_v3(terms_tr: str, ex_tr: str, terms_raw: str = "", ex_raw: str = "") -> str:
    """
    Rule-based, dataset-anchored labeling.
    Uses translated text first, falls back to raw text.
    Returns neutral topic labels (English).
    """
    t = _norm(f"{terms_tr} | {ex_tr}")
    r = _norm(f"{terms_raw} | {ex_raw}")

    def HAS(keys):
        return _has_any(t, keys) or _has_any(r, keys)

    if HAS(["sirens sounded", "red alert", "air raid siren", "incoming rocket", "rocket sirens", "alert in"]):
        return "Air raid sirens and alerts"

    if HAS(["intercepted", "interception", "air defense", "pvo", "aa defense", "downed uav", "shot down", "destroyed uav"]):
        return "Air defense interceptions"

    if HAS(["uav", "drone", "unmanned", "бпла", "дрон", "перехвачен", "сбит", "сбито", "над территорией", "средствами пво", "пво"]):
        return "UAV/drone incidents and interceptions"

    if HAS(["rosaviatsiya", "airport restrictions", "airport", "flight restrictions", "departure", "arrival", "аэропорт", "росавиация", "ограничения", "вылет", "прилет"]):
        return "Aviation restrictions and airport status"

    if HAS(["grouping of troops", "groupings of troops", "units", "brigade", "battalion", " фронт", "направлен",
            "подразделен", "группировка", "группировки войск", "министерство обороны", "mo d", "ministry of defense"]):
        return "Military operational updates"

    if HAS(["ministry of health", "gaza", "moh", "casualties", "killed", "injured", "martyrs", "ministry of health in gaza",
            "وزارة الصحة", "غزة", "شهيد", "شهداء", "اصابة", "الإصابات", "الاحتلال"]):
        return "Gaza health reporting and casualties"

    if HAS(["humanitarian", "aid", "relief", "famine", "malnutrition", "displaced", "tents", "shelter",
            "مساعدات", "اغاث", "مجاعة", "سوء تغذية", "نازحين", "خيام"]):
        return "Humanitarian conditions and aid"

    if HAS(["covid", "coronavirus", "cases", "daily cases", "total cases", "deaths", "recoveries",
            "كورونا", "إجمالي الحالات", "الحالات", "الوفيات", "متعاف"]):
        return "Epidemiological situation reporting (COVID)"

    if HAS(["prisoners", "hostages", "red cross", "remains", "bodies", "detained",
            "أسرى", "رهائن", "الصليب الأحمر", "جثامين", "جثث", "معتقل"]):
        return "Prisoners/hostages and remains reporting"

    base = (terms_tr or terms_raw or "").strip()
    if base:
        parts = [p.strip() for p in base.split(",") if p.strip()]
        if parts:
            return " / ".join(parts[:3]).title()

    return "General conflict-related updates"


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-IDF top terms per cluster for manual labeling (+ optional translation).")
    ap.add_argument("--run_dir", required=True, help="Clustering run folder (clustered_meta.csv, cluster_summary.csv)")
    ap.add_argument("--reports_root", default="reports", help="Root reports directory (versioned per run)")

    ap.add_argument("--min_cluster_size", type=int, default=30, help="Only label clusters with >= this size")
    ap.add_argument("--top_k_terms", type=int, default=12, help="Top terms per cluster")
    ap.add_argument("--max_clusters", type=int, default=80, help="Label at most top N largest clusters")
    ap.add_argument("--max_docs_per_cluster", type=int, default=1200, help="Cap docs per cluster for speed")
    ap.add_argument("--ngram_max", type=int, default=2, help="Use 1..ngram_max")

    ap.add_argument("--translate", action="store_true", help="Translate top terms/examples using googletrans")
    ap.add_argument("--translate_to", default="en", help="Target language code (default: en)")
    ap.add_argument("--translate_example", action="store_true", help="Also translate 1 representative example per cluster")
    ap.add_argument("--example_col", default="text_clean", help="Column to use for example text (default: text_clean)")
    ap.add_argument("--autolabel", action="store_true", help="Auto-fill draft_label using rules")
    ap.add_argument("--label_lang", default="en", help="Language of generated labels (en only)")

    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = os.path.join(args.reports_root, run_name)
    ensure_dir(out_dir)

    meta_path = os.path.join(args.run_dir, "clustered_meta.csv")
    summary_path = os.path.join(args.run_dir, "cluster_summary.csv")

    meta = pd.read_csv(meta_path)
    summary = pd.read_csv(summary_path)

    if "cluster" not in meta.columns or "text_clean" not in meta.columns:
        raise SystemExit("clustered_meta.csv must contain columns: cluster, text_clean")

    if args.example_col not in meta.columns:
        print(f"example_col '{args.example_col}' not found. Falling back to 'text_clean'.")
        args.example_col = "text_clean"

    meta = meta[meta["cluster"] != -1].copy()
    if meta.empty:
        raise SystemExit("No clustered rows found (all noise).")

    summary = summary.sort_values("size", ascending=False).copy()
    summary = summary[summary["size"] >= args.min_cluster_size].head(args.max_clusters)

    clusters = summary["cluster"].astype(int).tolist()
    print(f"Clusters selected for labeling: {len(clusters)} (min_size={args.min_cluster_size})")

    # Collect texts for TF-IDF
    texts = []
    doc_cluster = []

    for c in clusters:
        sub = meta[meta["cluster"] == c]
        if len(sub) > args.max_docs_per_cluster:
            sub = sub.sample(args.max_docs_per_cluster, random_state=42)
        t = sub["text_clean"].fillna("").astype(str).tolist()
        texts.extend(t)
        doc_cluster.extend([c] * len(t))

    print("Building TF-IDF...")
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        max_features=120000,
        ngram_range=(1, args.ngram_max),
        min_df=3,
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    doc_cluster = np.array(doc_cluster)

    print("Extracting top terms per cluster...")
    top_terms = {}
    for c in clusters:
        idx = np.where(doc_cluster == c)[0]
        if idx.size == 0:
            top_terms[c] = []
            continue

        Xc = X[idx]
        mean_scores = np.asarray(Xc.mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][: args.top_k_terms]
        terms = [(vocab[i], float(mean_scores[i])) for i in top_idx if mean_scores[i] > 0]
        top_terms[c] = terms

    print("Picking examples...")
    examples = {}
    has_views = "views" in meta.columns
    for c in clusters:
        sub = meta[meta["cluster"] == c].copy()
        if sub.empty:
            examples[c] = ""
            continue
        if has_views:
            sub["views"] = pd.to_numeric(sub["views"], errors="coerce").fillna(0)
            row = sub.sort_values("views", ascending=False).iloc[0]
        else:
            row = sub.iloc[0]
        examples[c] = str(row.get(args.example_col, "") or "")

    translator = None
    cache_path = os.path.join(out_dir, f"translate_cache_{args.translate_to}.json")
    if args.translate:
        translator = FreeGoogleTranslator(cache_path=cache_path, target=args.translate_to)

    print("Building label draft...")
    rows = []
    for _, r in summary.iterrows():
        c = int(r["cluster"])
        size = int(r["size"])

        conflict_dist = safe_json_loads(r.get("conflict_dist", "{}"))
        side_dist = safe_json_loads(r.get("side_dist", "{}"))
        type_dist = safe_json_loads(r.get("type_dist", "{}"))
        lang_top = safe_json_loads(r.get("lang_top", "{}"))

        terms_only = [t for t, _ in top_terms.get(c, [])]
        terms_joined = ", ".join(terms_only)

        terms_translated = ""
        if translator and terms_joined:
            terms_translated = translator.translate_text(terms_joined)

        ex = examples.get(c, "")
        ex_translated = ""
        if translator and args.translate_example and ex:
            ex_cut = ex[:800]
            ex_translated = translator.translate_text(ex_cut)

        draft_label = ""
        if args.autolabel:
            draft_label = autolabel_v3(
                terms_tr=terms_translated,
                ex_tr=ex_translated,
                terms_raw=terms_joined,
                ex_raw=ex,
            )

        rows.append({
            "cluster": c,
            "size": size,
            "top_terms": terms_joined,
            "top_terms_translated": terms_translated,
            "example": ex[:800],
            "example_translated": ex_translated,
            "draft_label": draft_label,
            "notes": "",
            "conflict_dist": json.dumps(conflict_dist, ensure_ascii=False),
            "side_dist": json.dumps(side_dist, ensure_ascii=False),
            "type_dist": json.dumps(type_dist, ensure_ascii=False),
            "lang_top": json.dumps(lang_top, ensure_ascii=False),
            "avg_views": r.get("avg_views", ""),
            "avg_forwards": r.get("avg_forwards", ""),
            "avg_replies": r.get("avg_replies", ""),
        })

    df_out = pd.DataFrame(rows).sort_values("size", ascending=False)

    draft_path = os.path.join(out_dir, "cluster_label_draft.csv")
    df_out.to_csv(draft_path, index=False, encoding="utf-8")

    terms_path = os.path.join(out_dir, "cluster_top_terms.json")
    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in top_terms.items()}, f, ensure_ascii=False, indent=2)

    if translator:
        translator.save()

    print("Done")
    print(f"Run dir        : {args.run_dir}")
    print(f"Reports dir    : {out_dir}")
    print(f"Draft labels   : {draft_path}")
    print(f"Top terms JSON : {terms_path}")
    if translator:
        print(f"Translate cache: {cache_path}")
    if args.translate_example:
        print("Included: example_translated column")


if __name__ == "__main__":
    main()
