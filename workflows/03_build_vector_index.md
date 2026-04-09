# Build Vector Index for RAG

## Objective
Turn each asset into an embedded document stored in ChromaDB so the chatbot can retrieve relevant past tests by semantic similarity + metadata filters.

## Inputs
- `.tmp/assets_with_text.json`

## Output
- `.tmp/chroma_db/` — persistent ChromaDB collection named `ad_assets`

## Document format
For each asset, build a single text document of the form:
```
OPENING: <opening hook text>

SCRIPT NAME: <script name>
GENRE: <genre> | IP: <IP name> | STYLE: <style>
WRITER: <writer> | DATE: <date>

PERFORMANCE:
- CPI: $X.XX (lower = better)
- CPM: $X.XX
- CTR*CTI: X.XX%
- Retention (0-95%): X.XX%
- Total test spend: $XXX
- Still active on growth: Yes/No

FULL SCRIPT:
<script text, truncated to 3000 chars>
```

This gives the embedding model rich context: the hook, the performance numbers, and the story content all in one chunk.

## Metadata (for Chroma filters)
Store alongside the embedding for later filtering:
- `ad_code`, `opening_code`, `writer`, `genre`, `ip`, `style`, `script_name`
- `cpi` (float), `cpm` (float), `ctr_cti` (float), `total_spend` (float)
- `is_active_growth` (bool) — derived from `Uploaded On Fb Growth`
- `date` (ISO string)

## Embedding model
Gemini `text-embedding-004` (1024 dims, free tier is plenty for 2.5k docs).

## Tool
`tools/build_index.py`

## Rebuild strategy
Rebuild the entire collection on each run — it's cheap for 2.5k docs and avoids stale entries. Add `--incremental` flag later if needed.
