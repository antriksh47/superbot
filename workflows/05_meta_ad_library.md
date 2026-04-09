# Meta Ad Library Integration (Phase 2)

## Status
**Not implemented yet — blocked on Meta developer access token.**

## Objective
Pull competitor short-drama ads running in the USA, extract their hooks, and feed them into the same vector index so the chatbot can benchmark against the broader market (not just our own tests). Also allow real-time lookup during chat ("what's ReelShort running on werewolf right now?").

## What we need from user
1. Meta developer app with Ad Library API access
2. Access token with `ads_read` scope
3. List of competitor page IDs to track (ReelShort, DramaBox, GoodShort, FlexTV, MoboReels, etc.)

## Planned tool
`tools/meta_ad_library.py` (stub exists)

## Planned flow
1. **Discovery**: query Ad Library search endpoint by country=US, industry/keywords = "drama", "werewolf", "romance"
2. **For each competitor page**: list all active ads, pull `ad_creative_body`, `ad_creative_link_caption`, impressions range, spend range, CTA
3. **Video extraction**: Ad Library exposes video URLs for active video ads — transcribe first 5 seconds with Whisper (or Gemini multimodal) to get the opening hook
4. **Store** in ChromaDB alongside our own assets with a `source=ad_library` tag so the chatbot can distinguish "our data" from "competitor signal"
5. **Refresh daily** via cron

## Real-time lookup
Add a `search_competitor_ads(genre, ip)` function the chatbot can call as a tool (Gemini function calling) when the user asks about current competitor activity.

## Important caveats
- Ad Library does NOT expose exact CPI/CPM for other advertisers — only spend RANGES (e.g. "$10k–$50k") and impression ranges. We cannot compare our CPIs to theirs directly. What we CAN learn: which hooks are running long (= working for them), which IPs are trending, what creative styles dominate.
- Ad Library API has strict rate limits. Batch and cache aggressively.
