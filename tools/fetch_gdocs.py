"""
Fetch Google Drive script/opening transcripts for each asset.
See workflows/02_fetch_google_docs.md for full spec.

First-time setup:
    python tools/fetch_gdocs.py --auth

Run:
    python tools/fetch_gdocs.py
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")

IN_PATH = Path(".tmp/assets.json")
OUT_PATH = Path(".tmp/assets_with_text.json")
CACHE_DIR = Path(".tmp/scripts_cache")


def get_drive_service():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_FILE.exists():
                raise SystemExit(
                    f"Missing {CREDS_FILE}. Download OAuth client credentials from "
                    "Google Cloud Console (Desktop app type) and save as credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            # open_browser=False so the URL is printed to stdout — useful when
            # running headless or when the auto-launched browser is the wrong profile.
            creds = flow.run_local_server(port=0, open_browser=False)
        TOKEN_FILE.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def fetch_file_text(service, file_id):
    """Return plain text for a Drive file_id. Handles Docs/docx/pdf/txt."""
    cache_path = CACHE_DIR / f"{file_id}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    try:
        meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    except HttpError as e:
        print(f"  ! get failed for {file_id}: {e.status_code}", file=sys.stderr)
        return None

    mime = meta.get("mimeType", "")
    name = meta.get("name", "")
    text = None

    try:
        if mime == "application/vnd.google-apps.document":
            data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
            text = data.decode("utf-8") if isinstance(data, bytes) else data

        elif mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ):
            buf = _download(service, file_id)
            try:
                import docx  # python-docx
            except ImportError:
                print("  ! python-docx not installed; skipping .docx", file=sys.stderr)
                return None
            doc = docx.Document(io.BytesIO(buf))
            text = "\n".join(p.text for p in doc.paragraphs)

        elif mime == "application/pdf":
            buf = _download(service, file_id)
            try:
                from pypdf import PdfReader
            except ImportError:
                print("  ! pypdf not installed; skipping .pdf", file=sys.stderr)
                return None
            reader = PdfReader(io.BytesIO(buf))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)

        elif mime.startswith("text/"):
            buf = _download(service, file_id)
            text = buf.decode("utf-8", errors="ignore")

        elif mime == "application/vnd.google-apps.folder":
            # Folder — grab first doc-like file inside
            resp = service.files().list(
                q=f"'{file_id}' in parents and trashed=false",
                fields="files(id,name,mimeType)",
                pageSize=20,
            ).execute()
            for f in resp.get("files", []):
                sub = fetch_file_text(service, f["id"])
                if sub:
                    text = sub
                    break

        else:
            print(f"  ! unsupported mime {mime} for {name}", file=sys.stderr)
            return None

    except HttpError as e:
        print(f"  ! fetch failed for {file_id} ({name}): {e.status_code}", file=sys.stderr)
        return None

    if text:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
    return text


def _download(service, file_id):
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", action="store_true", help="Run OAuth flow and exit")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N assets (for testing)")
    args = ap.parse_args()

    service = get_drive_service()
    if args.auth:
        print("Auth OK. token.json written.")
        return

    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found. Run ingest_csv.py first.")

    assets = json.loads(IN_PATH.read_text())
    if args.limit:
        assets = assets[: args.limit]

    # Collect unique file IDs
    unique_ids = set()
    for a in assets:
        if a.get("opening_transcript_file_id"):
            unique_ids.add(a["opening_transcript_file_id"])
        if a.get("script_transcript_file_id"):
            unique_ids.add(a["script_transcript_file_id"])

    print(f"Unique Drive file ids to fetch: {len(unique_ids)}")

    texts = {}
    for i, fid in enumerate(sorted(unique_ids), 1):
        print(f"[{i}/{len(unique_ids)}] {fid}")
        texts[fid] = fetch_file_text(service, fid)
        if i % 50 == 0:
            time.sleep(1)  # gentle pacing

    # Attach text to each asset
    for a in assets:
        a["opening_text"] = texts.get(a.get("opening_transcript_file_id")) or None
        a["script_text"] = texts.get(a.get("script_transcript_file_id")) or None

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(assets, indent=2, default=str))

    with_text = sum(1 for a in assets if a.get("script_text"))
    print(f"\nWrote {len(assets)} assets -> {OUT_PATH}")
    print(f"  with script text: {with_text}")
    print(f"  cached files: {len(list(CACHE_DIR.glob('*.txt'))) if CACHE_DIR.exists() else 0}")


if __name__ == "__main__":
    main()
