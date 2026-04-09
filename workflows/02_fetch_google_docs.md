# Fetch Google Doc Scripts

## Objective
For each asset, pull the full text of its `opening Transcript` and `Script transcript` Google Drive files and attach it to the asset record. This gives the RAG index the actual script content, not just the one-line hook.

## Inputs
- `.tmp/assets.json` — output of 01_ingest_csv
- Google OAuth credentials (`credentials.json`, `token.json`)

## Output
- `.tmp/assets_with_text.json` — same as assets.json but with `opening_text` and `script_text` fields populated
- `.tmp/scripts_cache/{file_id}.txt` — raw text cache keyed by Drive file_id (so reruns are cheap)

## Key behavior
- **Dedupe**: many rows share the same script URL. Cache by `file_id` and only fetch each unique ID once.
- **File type handling**: Drive files can be native Google Docs, .docx, or .pdf. Use Drive API `files.get` to check `mimeType`, then:
  - `application/vnd.google-apps.document` → `files.export(mimeType='text/plain')`
  - `application/vnd.openxmlformats-officedocument.wordprocessingml.document` → `files.get_media()` then parse with `python-docx`
  - `application/pdf` → `files.get_media()` then parse with `pypdf`
- **Graceful failure**: if a file can't be fetched (permissions, deleted), log it and leave the text field empty. Don't crash the whole run.
- **Rate limits**: Drive API free tier is generous but add small sleep every 50 requests to be safe.

## Tool
`tools/fetch_gdocs.py`

## First-run setup
User needs to run OAuth flow once:
```
python tools/fetch_gdocs.py --auth
```
This opens a browser, user signs in, and `token.json` is saved locally.
