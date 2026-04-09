# Ad Script RAG Chatbot

A data-backed chatbot that writes and critiques short-drama ad openings & scripts, grounded in your own historical testing data (CPI, CTR, retention, etc). Built on the WAT framework — see [CLAUDE.md](CLAUDE.md) for the architecture.

## What it does

- Ingests your Asset Testing Sheet (CSV) into clean JSON
- Pulls every linked Google Doc (opening transcripts + full scripts) via Drive API
- Embeds everything (opening + script + performance metrics) into a local ChromaDB vector store
- Runs a Streamlit chatbot on top that uses **Gemini 2.5 Pro + RAG** to propose hooks, rewrite drafts, and cite specific past winners by ad code and CPI
- Phase 2 (stub): Meta Ad Library integration to pull competitor ads into the same index

## Quick start

```bash
# 1. Install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.template .env
# Edit .env and add GOOGLE_API_KEY (Gemini) — get from https://aistudio.google.com/app/apikey

# 3. Google Drive OAuth (first time only)
# Download OAuth client credentials (Desktop app type) from Google Cloud Console,
# enable the Drive API on the project, save the file as credentials.json in repo root
python tools/fetch_gdocs.py --auth   # opens browser, writes token.json

# 4. Build the pipeline
python tools/ingest_csv.py                    # CSV -> .tmp/assets.json
python tools/fetch_gdocs.py                   # fetches all script Google Docs
python tools/build_index.py                   # embeds into .tmp/chroma_db

# 5. Run the chatbot
streamlit run tools/chatbot_app.py
```

If you want to try the chatbot before fetching Google Docs (faster first run):

```bash
python tools/ingest_csv.py
python tools/build_index.py --no-scripts
streamlit run tools/chatbot_app.py
```

The `--no-scripts` mode skips the full script text and indexes only the hook + metadata. Good enough to smoke-test. You'll get richer answers once the full scripts are indexed.

## Project layout

```
data/                        # Raw CSV export (gitignored)
workflows/                   # Markdown SOPs explaining each step
    01_ingest_csv.md
    02_fetch_google_docs.md
    03_build_vector_index.md
    04_run_chatbot.md
    05_meta_ad_library.md    # Phase 2
tools/                       # Python scripts
    ingest_csv.py
    fetch_gdocs.py
    build_index.py
    chatbot_app.py
    meta_ad_library.py       # Phase 2 stub
.tmp/                        # Intermediate outputs (gitignored)
    assets.json
    assets_with_text.json
    scripts_cache/{file_id}.txt
    chroma_db/
```

## What the chatbot understands

The system prompt teaches the model:
- Lower CPI = better (< $2.50 top-tier, $2.50–$3.50 good, > $4.00 weak)
- Higher CTR*CTI, higher 75–95% retention, and "scaled to Growth" flag all = winners
- Never invent ad codes — only cite retrieved tests
- When rewriting, explain WHY each change is data-backed

The sidebar supports filtering by genre, IP, style, writer, and CPI range.

## Phase 2: Meta Ad Library

See [workflows/05_meta_ad_library.md](workflows/05_meta_ad_library.md). Once you have a Meta developer token with `ads_read` scope, fill in `META_AD_LIBRARY_TOKEN` in `.env` and the list of competitor page IDs in `tools/meta_ad_library.py`. The Ad Library entries will be upserted into the same Chroma collection with a `source=ad_library` tag so the chatbot can distinguish competitor signal from your own data.

Important caveat: Ad Library does NOT expose exact CPIs from other advertisers — only spend/impression ranges. We can learn which hooks are running long (= working) but cannot directly compare CPIs.

## Notes

- The `.tmp/` directory is fully regenerable. Delete it and re-run the pipeline anytime.
- The Chroma collection rebuilds from scratch on each `build_index.py` run. This is intentional for simplicity at 2.5k docs.
- To refresh with new test results, drop a new CSV into `data/` and rerun the pipeline.
