"""
Build ChromaDB vector index from assets.
See workflows/03_build_vector_index.md for full spec.

Usage:
    python tools/build_index.py
    python tools/build_index.py --no-scripts   # skip if you haven't fetched gdocs yet
"""
import argparse
import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

CHROMA_DIR = Path(".tmp/chroma_db")
COLLECTION = "ad_assets"
EMBED_MODEL = "gemini-embedding-001"
# Paid tier 1 gets ~3000 RPM on gemini-embedding-001. Batches of 100 sail through.
BATCH = 100
PER_CALL_SLEEP_SEC = 0.0

# Prefer the enriched file if it exists
IN_WITH_TEXT = Path(".tmp/assets_with_text.json")
IN_BASIC = Path(".tmp/assets.json")


def build_document(a, include_script=True):
    perf_lines = []
    if a.get("cpi") is not None:
        perf_lines.append(f"- Actual CPI: ${a['cpi']:.2f} (lower is better)")
    if a.get("opening_lowest_cpi") is not None:
        perf_lines.append(f"- Opening's lowest CPI across all uses: ${a['opening_lowest_cpi']:.2f}")
    if a.get("cpm") is not None:
        perf_lines.append(f"- CPM: ${a['cpm']:.2f}")
    if a.get("ctr_cti") is not None:
        perf_lines.append(f"- CTR*CTI: {a['ctr_cti']:.2f}%")
    if a.get("completion_impression") is not None:
        perf_lines.append(f"- Completion/Impression: {a['completion_impression']:.2f}")
    if a.get("video_75_95") is not None:
        perf_lines.append(f"- Retention 75-95%: {a['video_75_95']:.2f}%")
    if a.get("total_spend") is not None:
        perf_lines.append(f"- Total test spend: ${a['total_spend']:.0f}")
    perf_lines.append(f"- Promoted to Growth (long-runner): {'yes' if a.get('is_active_growth') else 'no'}")

    doc_parts = [
        f"OPENING HOOK: {a.get('opening') or '(none)'}",
        "",
        f"SCRIPT NAME: {a.get('script_name') or '(unknown)'}",
        f"GENRE: {a.get('genre') or '?'} | IP: {a.get('ip') or '?'} | STYLE: {a.get('style') or '?'}",
        f"WRITER: {a.get('writer') or '?'} | TEST DATE: {a.get('test_date_raw') or '?'}",
        f"AD CODE: {a.get('ad_code') or '?'} | OPENING CODE: {a.get('opening_code') or '?'}",
        "",
        "PERFORMANCE:",
        *perf_lines,
    ]
    # Include both opening + script when available, both with generous limits
    # gemini-embedding-001 supports ~2048 tokens (~8k chars), so we stay under that.
    if include_script and a.get("opening_text"):
        doc_parts.append("")
        doc_parts.append("OPENING TRANSCRIPT:")
        doc_parts.append(a["opening_text"][:2500])
    if include_script and a.get("script_text"):
        doc_parts.append("")
        doc_parts.append("FULL SCRIPT:")
        doc_parts.append(a["script_text"][:5500])
    return "\n".join(doc_parts)


def build_metadata(a):
    """Chroma only allows scalar metadata values."""
    def clean(v):
        if v is None:
            return ""
        if isinstance(v, (int, float, bool, str)):
            return v
        return str(v)

    return {
        "ad_code": clean(a.get("ad_code")),
        "opening_code": clean(a.get("opening_code")),
        "writer": clean(a.get("writer")),
        "genre": clean(a.get("genre")),
        "ip": clean(a.get("ip")),
        "style": clean(a.get("style")),
        "script_name": clean(a.get("script_name")),
        "cpi": float(a["cpi"]) if a.get("cpi") is not None else -1.0,
        "cpm": float(a["cpm"]) if a.get("cpm") is not None else -1.0,
        "ctr_cti": float(a["ctr_cti"]) if a.get("ctr_cti") is not None else -1.0,
        "total_spend": float(a["total_spend"]) if a.get("total_spend") is not None else 0.0,
        "is_active_growth": bool(a.get("is_active_growth")),
        "date": clean(a.get("test_date_raw")),
        "opening": clean(a.get("opening"))[:500],
    }


def embed_one(client, text):
    """Single-item embed with retry + 429 backoff."""
    text = text[:7000]  # gemini-embedding-001 max input ≈ 2048 tokens
    for attempt in range(8):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return resp.embeddings[0].values
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 30 + attempt * 15
            else:
                wait = 2 ** attempt
            print(f"  ! embed retry {attempt+1} in {wait}s: {msg[:100]}", flush=True)
            time.sleep(wait)
    raise RuntimeError("embedding failed after retries")


def embed_batch(client, texts):
    """Embed texts one-by-one with throttling to stay under free-tier rate limits."""
    out = []
    for t in texts:
        out.append(embed_one(client, t))
        time.sleep(PER_CALL_SLEEP_SEC)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-scripts", action="store_true", help="Skip script text, use basic assets.json")
    ap.add_argument("--limit", type=int, default=None, help="Only index first N assets (smoke test)")
    ap.add_argument("--top-by-cpi", type=int, default=None, help="Index N best-CPI assets only")
    ap.add_argument("--resume", action="store_true", help="Keep existing collection, only embed missing assets")
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY missing. Set it in .env")

    in_path = IN_BASIC if args.no_scripts or not IN_WITH_TEXT.exists() else IN_WITH_TEXT
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found. Run ingest_csv.py first.")
    print(f"Reading {in_path}")

    assets = json.loads(in_path.read_text())
    # Only keep assets with an opening
    assets = [a for a in assets if a.get("opening")]
    if args.top_by_cpi:
        with_cpi = [a for a in assets if a.get("cpi") is not None]
        with_cpi.sort(key=lambda a: a["cpi"])
        assets = with_cpi[: args.top_by_cpi]
    elif args.limit:
        assets = assets[: args.limit]
    print(f"Indexing {len(assets)} assets")

    gclient = genai.Client(api_key=api_key)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if args.resume:
        # Keep existing collection; we'll skip assets whose IDs are already in it.
        try:
            coll = chroma.get_collection(COLLECTION)
            existing_ids = set(coll.get(include=[])["ids"])
            print(f"Resume: collection has {len(existing_ids)} existing docs")
        except Exception:
            coll = chroma.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
            existing_ids = set()
            print("Resume: collection didn't exist, starting fresh")
    else:
        # Nuke and rebuild
        try:
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass
        coll = chroma.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        existing_ids = set()

    total = 0
    skipped = 0
    for i in range(0, len(assets), BATCH):
        batch = assets[i : i + BATCH]
        docs = [build_document(a, include_script=not args.no_scripts) for a in batch]
        metas = [build_metadata(a) for a in batch]
        ids = [f"asset_{a.get('sr_no') or i+j}_{a.get('ad_code') or 'x'}" for j, a in enumerate(batch)]
        # de-dupe ids
        seen = set()
        uniq_ids = []
        for _id in ids:
            base = _id
            k = 1
            while _id in seen:
                _id = f"{base}_{k}"
                k += 1
            seen.add(_id)
            uniq_ids.append(_id)

        # Resume: filter out anything already in the collection
        if existing_ids:
            keep_idx = [j for j, _id in enumerate(uniq_ids) if _id not in existing_ids]
            if not keep_idx:
                skipped += len(batch)
                total += len(batch)
                print(f"  skipped {len(batch)} (already indexed) — {total}/{len(assets)}", flush=True)
                continue
            if len(keep_idx) < len(uniq_ids):
                skipped += len(uniq_ids) - len(keep_idx)
            uniq_ids = [uniq_ids[j] for j in keep_idx]
            docs = [docs[j] for j in keep_idx]
            metas = [metas[j] for j in keep_idx]

        embeds = embed_batch(gclient, docs)
        coll.add(ids=uniq_ids, documents=docs, metadatas=metas, embeddings=embeds)
        total += len(batch)
        print(f"  indexed {total}/{len(assets)}", flush=True)

    msg = f"\nDone. Collection '{COLLECTION}' has {coll.count()} documents at {CHROMA_DIR}"
    if skipped:
        msg += f" (skipped {skipped} already-indexed in this run)"
    print(msg)


if __name__ == "__main__":
    main()
