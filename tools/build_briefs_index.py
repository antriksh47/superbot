"""
Build a ChromaDB collection of show brief chunks (hub briefs + linked 10HR
base story docs + cracking-script docs). This is a separate collection from
the ad-level index so the chatbot can retrieve both independently.

Usage:
    python tools/build_briefs_index.py
    python tools/build_briefs_index.py --resume
"""
import argparse
import json
import os
import re
import socket
import time
from pathlib import Path

# Force a global socket timeout — without this, hung connections to the Gemini
# embed endpoint can wedge the build forever instead of failing fast and retrying.
socket.setdefaulttimeout(60)

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

CHROMA_DIR = Path(".tmp/chroma_db")
COLLECTION = "show_briefs"
EMBED_MODEL = "gemini-embedding-001"
BATCH = 100

IN_PATH = Path(".tmp/show_briefs.json")

# Chunking
CHUNK_SIZE = 1800          # chars per chunk
CHUNK_OVERLAP = 300

# Exclusion — files that are non-story (subtitles, audio filenames, etc.)
SKIP_PATTERNS = re.compile(
    r"^(base\.srt|ps\s*\d.*\.srt|.*_srt_file|.*\.mp[34]|audio|ps\d|base[-_ ]?raw|mix|srt sheet|link)$",
    re.I,
)
MIN_CONTENT_CHARS = 800    # skip linked docs smaller than this


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, biased to break on paragraph/sentence boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        end = i + size
        if end >= len(text):
            chunks.append(text[i:])
            break
        # Try to break at nearest paragraph within last 300 chars
        window = text[max(i, end - 300):end]
        pp = window.rfind("\n\n")
        sp = window.rfind(". ")
        if pp >= 0:
            end = max(i, end - 300) + pp + 2
        elif sp >= 0:
            end = max(i, end - 300) + sp + 2
        chunks.append(text[i:end].strip())
        i = end - overlap
    return [c for c in chunks if c.strip()]


def classify_doc_type(name, anchor):
    """Rough guess at what a linked doc is, for metadata."""
    s = (name or "") + " " + (anchor or "")
    s = s.lower()
    if "10hr" in s or "10 hr" in s or "cms script" in s or "base story" in s:
        return "base_story"
    if "rework" in s or "rewrite" in s:
        return "cpi_crack_script"
    if "character canvas" in s or "_cc" in s or " cc" in s:
        return "character_canvas"
    if "brief" in s:
        return "brief_hub"
    if ".srt" in s or "subtitle" in s:
        return "subtitle"
    return "other"


def embed_batch(client, texts):
    texts = [t[:7500] for t in texts]
    for attempt in range(6):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:
            msg = str(e)
            wait = 30 + attempt * 15 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 2 ** attempt
            print(f"  ! embed retry in {wait}s: {msg[:100]}", flush=True)
            time.sleep(wait)
    raise RuntimeError("embed failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY missing")

    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} missing. Run tools/fetch_briefs.py first.")
    shows = json.loads(IN_PATH.read_text())
    print(f"Loaded {len(shows)} show packages")

    # Build chunks
    chunks = []
    skipped_small = 0
    skipped_name = 0
    for show in shows:
        slug = show["show_slug"]
        show_name = show.get("name") or show.get("anchor_text") or slug

        # Hub brief — always include, one chunk
        hub_text = show.get("hub_text") or ""
        if hub_text.strip():
            for j, c in enumerate(chunk_text(hub_text)):
                chunks.append({
                    "id": f"brief_{slug}_hub_{j}",
                    "text": c,
                    "meta": {
                        "show_slug": slug,
                        "show_name": show_name,
                        "doc_type": "brief_hub",
                        "doc_name": show_name,
                        "source_file_id": show["file_id"],
                        "chunk_idx": j,
                    },
                })

        # Linked docs
        for linked in show.get("linked_docs", []):
            lname = linked.get("name") or linked.get("anchor_text") or ""
            ltext = linked.get("text") or ""
            if SKIP_PATTERNS.match(lname.strip()):
                skipped_name += 1
                continue
            if len(ltext) < MIN_CONTENT_CHARS:
                skipped_small += 1
                continue
            doc_type = classify_doc_type(lname, linked.get("anchor_text"))
            # Skip pure subtitles — they tend to be noisy
            if doc_type == "subtitle":
                skipped_name += 1
                continue
            for j, c in enumerate(chunk_text(ltext)):
                chunks.append({
                    "id": f"brief_{slug}_{linked['file_id']}_{j}",
                    "text": c,
                    "meta": {
                        "show_slug": slug,
                        "show_name": show_name,
                        "doc_type": doc_type,
                        "doc_name": lname[:200],
                        "source_file_id": linked["file_id"],
                        "chunk_idx": j,
                    },
                })

    print(f"Built {len(chunks)} chunks")
    print(f"  skipped (small): {skipped_small}")
    print(f"  skipped (name/subtitle): {skipped_name}")
    by_type = {}
    for c in chunks:
        by_type[c["meta"]["doc_type"]] = by_type.get(c["meta"]["doc_type"], 0) + 1
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n}")

    gclient = genai.Client(api_key=api_key)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if args.resume:
        try:
            coll = chroma.get_collection(COLLECTION)
            existing = set(coll.get(include=[])["ids"])
            print(f"Resume: {len(existing)} existing")
        except Exception:
            coll = chroma.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
            existing = set()
    else:
        try:
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass
        coll = chroma.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        existing = set()

    total = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        batch = [c for c in batch if c["id"] not in existing]
        if not batch:
            total += BATCH
            continue
        docs = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metas = [c["meta"] for c in batch]
        embeds = embed_batch(gclient, docs)
        coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
        total += len(batch)
        print(f"  indexed {total}/{len(chunks)}", flush=True)

    print(f"\nDone. '{COLLECTION}' now has {coll.count()} chunks")


if __name__ == "__main__":
    main()
