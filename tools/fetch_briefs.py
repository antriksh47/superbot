"""
Fetch show-level Script Brief docs from Google Drive PLUS one level of
linked sub-docs (10hr base story, CPI crack notes, character canvas),
and build a per-show knowledge package the chatbot uses for deep context.

Approach:
1. Load brief file IDs from .tmp/ndnf_hyperlinks.json
2. For each brief, export as HTML and extract all Drive hyperlinks inside
3. Fetch each unique linked Drive doc (one level deep only, no infinite recursion)
4. Assemble a show package: {slug, name, hub_text, linked_docs: [...]}

Output: .tmp/show_briefs.json
Caches: .tmp/briefs_cache/ (brief hub text), .tmp/briefs_linked_cache/ (linked docs)

Usage:
    python tools/fetch_briefs.py
"""
import io
import json
import re
import sys
import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = Path("token.json")

HYPERLINKS_PATH = Path(".tmp/ndnf_hyperlinks.json")
CACHE_DIR = Path(".tmp/briefs_cache")
LINKED_CACHE_DIR = Path(".tmp/briefs_linked_cache")
OUT_PATH = Path(".tmp/show_briefs.json")

# Max number of sub-docs to fetch per brief (safety limit)
MAX_LINKED_PER_BRIEF = 12


def get_drive_service():
    if not TOKEN_FILE.exists():
        raise SystemExit(
            f"Missing {TOKEN_FILE}. Run 'python tools/fetch_gdocs.py --auth' first "
            "to authenticate with Google Drive."
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json())
        else:
            raise SystemExit(
                f"Token expired and can't be refreshed. Re-run "
                "'python tools/fetch_gdocs.py --auth' to re-authenticate."
            )
    return build("drive", "v3", credentials=creds)


def slugify(name):
    """Turn 'Script Brief - TAB : Broken Necklace' -> 'tab_broken_necklace'."""
    s = re.sub(r"(?i)^\s*script\s*brief\s*[-:]?\s*", "", name)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "unknown"


DRIVE_ID_RE = re.compile(r"/(?:document|spreadsheets|file|presentation)/d/([a-zA-Z0-9_-]{20,})|[?&]id=([a-zA-Z0-9_-]{20,})")


def extract_drive_ids_from_html(html):
    """Return list of (file_id, anchor_text) from Drive links in HTML."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  ! beautifulsoup4 missing", file=sys.stderr)
        return []
    soup = BeautifulSoup(html, "html.parser")
    ids = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Google wraps external links as redirect: https://www.google.com/url?q=...
        if "url?q=" in href:
            import urllib.parse
            href = urllib.parse.unquote(href.split("url?q=", 1)[1].split("&")[0])
        m = DRIVE_ID_RE.search(href)
        if m:
            fid = m.group(1) or m.group(2)
            ids.append((fid, (a.get_text() or "").strip()[:100]))
    return ids


def fetch_hub(service, file_id, cache_dir):
    """Fetch a hub brief: returns (name, mime, plain_text, html)."""
    cache_txt = cache_dir / f"{file_id}.txt"
    cache_html = cache_dir / f"{file_id}.html"
    if cache_txt.exists() and cache_html.exists():
        # Peek meta from cache only — name unknown, so we re-probe
        try:
            meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
            return meta.get("name"), meta.get("mimeType"), cache_txt.read_text(), cache_html.read_text()
        except HttpError:
            return None, None, cache_txt.read_text(), cache_html.read_text()
    try:
        meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    except HttpError as e:
        print(f"  ! meta failed {file_id}: {e}", file=sys.stderr)
        return None, None, None, None
    mime = meta.get("mimeType", "")
    name = meta.get("name", "")
    text = None
    html = None
    try:
        if mime == "application/vnd.google-apps.document":
            text_bytes = service.files().export(fileId=file_id, mimeType="text/plain").execute()
            text = text_bytes.decode("utf-8") if isinstance(text_bytes, bytes) else text_bytes
            html_bytes = service.files().export(fileId=file_id, mimeType="text/html").execute()
            html = html_bytes.decode("utf-8") if isinstance(html_bytes, bytes) else html_bytes
        elif mime == "application/vnd.google-apps.spreadsheet":
            # Export as CSV just to get flattened text — also HTML for hyperlinks
            text_bytes = service.files().export(fileId=file_id, mimeType="text/csv").execute()
            text = text_bytes.decode("utf-8") if isinstance(text_bytes, bytes) else text_bytes
            try:
                html_bytes = service.files().export(fileId=file_id, mimeType="text/html").execute()
                html = html_bytes.decode("utf-8") if isinstance(html_bytes, bytes) else html_bytes
            except HttpError:
                html = ""
        else:
            _, text = fetch_linked(service, file_id, None)  # delegate
            html = ""
    except HttpError as e:
        print(f"  ! fetch failed {file_id} ({name}): {e}", file=sys.stderr)
        return name, mime, None, None

    if text:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_txt.write_text(text)
    if html is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_html.write_text(html)
    return name, mime, text, html


def fetch_linked(service, file_id, anchor):
    """Fetch a linked sub-doc: returns (name, text). Cached."""
    cache = LINKED_CACHE_DIR / f"{file_id}.txt"
    if cache.exists():
        return anchor or "(cached)", cache.read_text()
    try:
        meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    except HttpError as e:
        print(f"    ! linked meta failed {file_id}: {str(e)[:80]}", file=sys.stderr)
        return None, None
    mime = meta.get("mimeType", "")
    name = meta.get("name", "")
    text = None
    try:
        if mime == "application/vnd.google-apps.document":
            data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
            text = data.decode("utf-8") if isinstance(data, bytes) else data
        elif mime == "application/vnd.google-apps.spreadsheet":
            data = service.files().export(fileId=file_id, mimeType="text/csv").execute()
            text = data.decode("utf-8") if isinstance(data, bytes) else data
        elif mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ):
            buf = _download(service, file_id)
            try:
                import docx
                doc = docx.Document(io.BytesIO(buf))
                text = "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                pass
        elif mime == "application/pdf":
            buf = _download(service, file_id)
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(buf))
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                pass
        elif mime.startswith("text/"):
            text = _download(service, file_id).decode("utf-8", errors="ignore")
        elif mime == "application/vnd.google-apps.folder":
            return name, None  # skip folders
        else:
            return name, None
    except HttpError as e:
        print(f"    ! linked fetch failed {file_id} ({name}): {str(e)[:80]}", file=sys.stderr)
        return name, None

    if text:
        LINKED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return name, text


def _download(service, file_id):
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def main():
    if not HYPERLINKS_PATH.exists():
        raise SystemExit(
            f"{HYPERLINKS_PATH} missing. First run the NDNF tracker extractor "
            "to dump all sheet hyperlinks."
        )

    hyperlinks = json.loads(HYPERLINKS_PATH.read_text())
    # Keep only links that look like script briefs — either from brief sheets or with
    # "brief" in the anchor text.
    brief_links = [
        h for h in hyperlinks
        if "brief" in (h.get("cell_text") or "").lower()
        or "brief" in (h.get("sheet") or "").lower()
    ]
    # Dedupe by file_id, keep first sheet+cell_text seen
    seen = {}
    for h in brief_links:
        fid = h["file_id"]
        if fid not in seen:
            seen[fid] = h
    print(f"Unique brief file ids to fetch: {len(seen)}")

    service = get_drive_service()
    records = []
    fetched_linked_global = {}  # file_id -> (name, text) — dedup across briefs
    total_linked_attempted = 0
    total_linked_fetched = 0

    for i, (fid, h) in enumerate(sorted(seen.items()), 1):
        anchor = h.get("cell_text") or ""
        print(f"[{i}/{len(seen)}] HUB {fid}  {anchor[:60]}")
        name, mime, hub_text, hub_html = fetch_hub(service, fid, CACHE_DIR)
        if not hub_text:
            continue

        # Extract linked Drive IDs from hub HTML
        linked_ids = []
        if hub_html:
            linked_ids = extract_drive_ids_from_html(hub_html)
            # dedupe + exclude self + cap
            seen_linked = set()
            linked_ids = [(x, t) for x, t in linked_ids if x != fid and not (x in seen_linked or seen_linked.add(x))][:MAX_LINKED_PER_BRIEF]

        linked_docs = []
        for lid, ltext in linked_ids:
            total_linked_attempted += 1
            if lid in fetched_linked_global:
                lname, ldata = fetched_linked_global[lid]
            else:
                print(f"    -> sub {lid} ({ltext[:50]})")
                lname, ldata = fetch_linked(service, lid, ltext)
                fetched_linked_global[lid] = (lname, ldata)
                time.sleep(0.2)
            if ldata:
                total_linked_fetched += 1
                linked_docs.append({
                    "file_id": lid,
                    "name": lname,
                    "anchor_text": ltext,
                    "text": ldata,
                    "char_count": len(ldata),
                })

        combined_chars = len(hub_text) + sum(d["char_count"] for d in linked_docs)
        records.append({
            "file_id": fid,
            "name": name,
            "anchor_text": anchor,
            "show_slug": slugify(name or anchor),
            "hub_text": hub_text,
            "hub_char_count": len(hub_text),
            "linked_docs": linked_docs,
            "total_char_count": combined_chars,
            "sheet": h.get("sheet"),
        })
        if i % 10 == 0:
            time.sleep(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} show packages -> {OUT_PATH}")
    print(f"Hub cache: {len(list(CACHE_DIR.glob('*.txt')))} files")
    print(f"Linked cache: {len(list(LINKED_CACHE_DIR.glob('*.txt')))} files")
    print(f"Linked docs attempted: {total_linked_attempted} / fetched: {total_linked_fetched}")
    total_chars = sum(r["total_char_count"] for r in records)
    print(f"Total combined text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    print(f"Slugs: {sorted(set(r['show_slug'] for r in records))}")


if __name__ == "__main__":
    main()
