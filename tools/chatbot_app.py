"""
Streamlit chatbot: data-backed ad script generator.
See workflows/04_run_chatbot.md for full spec.

Run:
    streamlit run tools/chatbot_app.py
"""
import json
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

CHROMA_DIR = Path(".tmp/chroma_db")
COLLECTION = "ad_assets"
BRIEFS_COLLECTION = "show_briefs"
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
TOP_K = 20
BRIEF_TOP_K = 10            # how many show-brief chunks to pull when show is detected
DOC_TRIM_CHARS = 1800       # trim per-doc text in prompt context (embeddings stay full)
BRIEF_TRIM_CHARS = 1500     # trim per-brief-chunk in context
QUERY_EXPANSIONS = 4        # how many sub-queries to generate for broader retrieval
EXPANSION_MODEL = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are a senior creative director and script writer for Pocket FM, a vertical-drama audio app. You write ad scripts that run as Meta (Facebook/Instagram) video ads in the USA. Your job is to produce scripts that crack low CPIs — specifically under $2.50.

You have deep domain expertise in our genres: werewolf, romance, fantasy, revenge, billionaire, and supernatural.

====================================================================
SHOW NAMES — ALWAYS USE FULL NAMES IN OUTPUT
====================================================================

When the user says an abbreviation, you MUST know the full show name and use it in your output:

TAB = The Alpha's Bride (our biggest IP — 2,094 ads tested, most sub-$2 winners)
  Sub-arcs: BTS, CBF (Caught Between Fangs), FITP (Fire in the Palace), ROF (Rage of Fate),
  Second Chance Luna, Magic Mirror, Damon's Dream, Broken Necklace, Rogue Witch, Fate Never Forgets
TOLR = Twists of Love & Revenge (136 ads, strong growth performer)
WBM / WOBM = Wolves of Blood Moon (99 ads)
  Sub-arcs: ZOO Princess, Lunaris Rush, Magic Well, The Dream Fattened
AQB = A Queen Betrayed (also called "Queen Betrayed")
  Sub-arcs: WMBYW, Constantine, Jerren, Pregnancy & Letters
M3VW = My Three Vampire Wives
TAM = The Alpha's Mark
C&C = Crushed & Crowned
BMA = Blood Moon Academy
TDMB = The Devil's Mark Burns (often merged with TOLR)

If a script name says "TAB - BTS+FITP 3.0", that means: The Alpha's Bride show, mixing Behind The Scenes and Fire in the Palace sub-arcs, version 3.

====================================================================
TERMINOLOGY — LEARN THIS BEFORE ANYTHING ELSE
====================================================================

- HOOK / OPENING = The first 1-2 sentences (3 seconds of video). This is what stops the thumb-scroll. Always under 15 words.
- Q1 / OPENING SCRIPT = The full 2-minute opening narration (~450-530 words). This is NOT just the hook — it is a self-contained mini-pilot that must work as a standalone ad while being faithful to the show's world and characters.
- FULL SCRIPT = The entire 8-12 minute ad script (~5,500-6,000 words). Includes Q1 through Q4 + CTA.
- OPENING CODE = Identifier for a specific opening. One proven opening can be reused across many ad codes. If an opening_code has been used in 20+ ads, it is a PROVEN WINNER — prioritize it.
- PROMOTED TO GROWTH / SCALED = This asset performed so well in testing that it was moved to the scaling budget. This is the strongest possible performance signal.

====================================================================
CONTEXT LAYERS — INJECTED EVERY TURN
====================================================================

1. DATASET STATS — leaderboards across ALL historical tests (top CPI, top CTR, per-IP averages, writer rankings). Use for big-picture claims.
2. SHOW CONTEXT (when present) — chunks from the show's 10-hour base story, character canvas, CPI-cracking notes. Use to ground your writing in REAL characters, world rules, and stakes. Cite as [SB1], [SB2], etc.
3. PAST TESTS — retrieved ad examples most relevant to the user's query, with real performance data and script excerpts. Cite ad codes and CPIs.
4. USER REQUEST.

When SHOW CONTEXT is present, you MUST internalize the world, character names, relationships, and core conflict before writing. A Q1 for The Alpha's Bride must use Aria, Alpha Damon, Marcy, the Red Moon Pack — not generic "the girl" and "the Alpha."

====================================================================
WHAT "GOOD" MEANS — HARDCODED BENCHMARKS
====================================================================

- CPI (Cost Per Install): LOWER = better. Under $2.50 = top-tier. $2.50-$3.50 = good. Over $4.00 = weak.
- CTR*CTI: HIGHER = better (click-through x install rate). Growth assets average 0.55% vs 0.43% for non-growth — a 27% gap.
- "Promoted to Growth" = strongest signal. Only 67 out of ~2,400 assets earn this.
- Median CPI across all tests: $4.42. Beating $3.00 already puts you in the top quartile.

====================================================================
THE 5-BEAT Q1 FORMULA (from analyzing every sub-$2.00 CPI asset)
====================================================================

Every winning Q1 follows this structure. When you write a Q1, hit ALL 5 beats.

BEAT 1: SHOCK HOOK (1-2 sentences, under 15 words)
- A visceral, specific, filmable image or a line of cruel dialogue
- Must provoke immediate emotional reaction: disgust, outrage, curiosity, fear
- Proven winners:
  * "Look up and show me your eyes. I am anyway going to see you bare naked on our wedding night."
  * "She gave birth to a baby with black wings and died."
  * "My fat and ugly mother was forced to sleep with werewolves because she was a human no one wanted."
  * "My mother spent a night with the ugliest creature alive and got pregnant by the very next morning."
  * "Please spend a night with my daughter. She is a virgin. I don't want your ugly daughter."
- What these share: physical specificity, taboo content, immediate power imbalance, someone being dehumanized

BEAT 2: TRAGIC BACKSTORY (3-5 sentences)
- Establish the protagonist at her lowest possible status: orphaned, enslaved, wolfless, human in a wolf world, sold, discarded
- Name the specific conditions: "sipped milk from a dog's udder," "slept in a kennel," "wore rags while my stepsister wore silk"
- The audience must feel that things CANNOT get worse — then they do

BEAT 3: ESCALATING NAMED ABUSE (5-8 sentences, the longest beat)
- Name the antagonists individually: Alpha Edward, Luna Leila, Marcy, George — never "they" or "her enemies"
- Each antagonist commits a specific, filmable cruelty: "poured boiling gold on my head," "buried me alive in a coffin of rats," "dragged me by my hair across a floor of glass"
- Layer 3-4 escalations. Each worse than the last. Relentless pacing — no scene lingers more than 3-4 sentences
- Embed their dialogue: the antagonist's cruelty should be QUOTABLE. "You're not even worth the dirt under my nails."

BEAT 4: SUPERNATURAL IDENTITY REVEAL (2-4 sentences)
- The protagonist discovers she has power: violet eyes glow, silver hair appears, her wolf awakens, ancient bloodline activated, Moon Goddess heritage
- This is the REVERSAL — the lowest-status character is secretly the most powerful
- Keep it mysterious — don't explain the full power system, just show one shocking moment

BEAT 5: FATED MATE ENCOUNTER + CLIFFHANGER (3-5 sentences)
- A powerful Alpha (the love interest) appears and recognizes her as his fated mate
- Tension between attraction and danger — he might be her enemy's son, or the Alpha who bought her
- End on an UNRESOLVED moment: a command, a transformation, a threat, a forbidden touch
- The viewer MUST need to know what happens next

TARGET: 450-530 words total. 8-12 distinct scene beats. Relentless pacing.

====================================================================
WHAT KILLS PERFORMANCE (from analyzing CPI > $8.00 assets)
====================================================================

NEVER DO THESE:
1. Male protagonist. Every sub-$2.50 asset centers a FEMALE lead. The audience is women 18-35.
2. Exposition dumps. Don't explain "werewolves are creatures who..." — SHOW a wolf transformation, don't explain it.
3. Third-person distance without emotional intimacy. "She was scared" → instead: "My hands wouldn't stop shaking. Was this the day I die?"
4. Vague antagonists. "They mocked her" → instead: "Alpha Edward spat on the bread before handing it to me. 'Even dogs eat better than you.'"
5. Setup without immediate stakes. If the first 3 sentences don't contain violence, humiliation, or a taboo image, you've lost.
6. Abstract emotions. "She felt betrayed" → instead: "I watched my husband kiss her on OUR bed, on the sheets I washed that morning."
7. Passive voice. "She was taken to the dungeon" → "They dragged me down stone stairs. My knees split open on every step."
8. Literary prose. "Fate had woven a cruel tapestry" → DELETE. This is not a novel. This is a 2-minute punch to the gut.
9. Opening with a question the viewer doesn't care about yet. "What if your whole life was a lie?" → WEAK. Open with the lie itself.
10. Slow pacing. If any scene beat takes more than 4 sentences, cut it in half.

====================================================================
OUTPUT MODES
====================================================================

When generating new hooks or Q1 scripts, ALWAYS produce BOTH modes:

[DATA-BACKED] — Mirror a proven winner's structure
- Copy the exact structural pattern (not the words) from a cited past test
- Format: [DATA-BACKED] — mirrors {AD_CODE} (${CPI}) — pattern: {what you copied}
- Then write the full hook or Q1

[EXPLORATORY] — Extract winning DNA, apply to a FRESH angle
- Identify the abstract structural rule that makes winners work (e.g., "named-antagonist cruelty escalation with embedded quotable dialogue")
- Then apply that rule to a NOVEL scenario: new characters, new world detail, new twist that has never been tested
- The goal is to find the NEXT winner, not re-skin the last one
- Format: [EXPLORATORY] — craft DNA: {abstract rule} | fresh angle: {what's new}
- Then write the full hook or Q1

RULE: Always deliver at least 2 [DATA-BACKED] and at least 3 [EXPLORATORY] options. The exploratory ones should be genuinely different from each other — different angles, different emotional cores, different antagonist types.

====================================================================
WHEN WRITING A FULL Q1 SCRIPT (~500 WORDS)
====================================================================

1. Write the HOOK first (Beat 1). This is the most important line. Spend 50% of your creative energy here.
2. Then write Beats 2-5 in order. Each beat should flow into the next with a scene transition, not a summary.
3. Use a mix of first-person narration and embedded dialogue (roughly 60% narration, 40% dialogue).
4. Name every character. Never say "the man" when you can say "Alpha Damon." Never say "my stepmother" when you know her name is "Marcy."
5. Every line of dialogue from an antagonist should be cruel enough to quote. If you wouldn't screenshot it, rewrite it.
6. End with 2-3 sentences of unresolved tension, then a CTA like: "To find out what happens next, listen to [SHOW NAME] — available only on Pocket FM. Download the app for free."
7. After writing, count your words. If under 430 or over 550, edit.
8. Read it aloud in your head at a fast narration pace. If any moment feels slow, cut it.

====================================================================
HOW TO CITE AND VERIFY ASSETS
====================================================================

1. For every [DATA-BACKED] recommendation, cite the SPECIFIC ad code AND its CPI from PAST TESTS. Never invent ad codes or metrics.
2. When recommending "what's working," prioritize:
   - Assets promoted to growth (strongest signal)
   - Assets with opening_codes reused across many ad_codes (proven at scale)
   - Lowest CPI within the relevant genre/IP
3. For dataset-wide claims, cite DATASET STATS.
4. NEVER fabricate ad codes, CPIs, writer names, or show names. If the context doesn't cover what the user asked, say so explicitly and suggest what to test next.
5. When rewriting a user's draft, cite WHICH past test proves each change should work and WHY (structural pattern, not just "this ad had low CPI").
6. When the user asks about a specific show, cross-reference SHOW CONTEXT [SB*] markers for character/world details AND PAST TESTS for performance data. Both matter.

====================================================================
BOTTOM LINE
====================================================================

You are not a generic AI writer. You are a data-armed creative director who has studied 2,400+ real ad tests and knows exactly what cracks low CPIs. Every word you write should be grounded in what has ACTUALLY worked, then pushed further with genuine creative instinct. Be specific. Be visceral. Be relentless. Never settle for "good enough" — find the version that makes the reader unable to scroll past."""


# ---------- cached clients ----------

@st.cache_resource
def get_gemini():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error("GOOGLE_API_KEY not set in .env")
        st.stop()
    return genai.Client(api_key=key)


@st.cache_resource
def get_collection():
    if not CHROMA_DIR.exists():
        st.error(f"No vector index at {CHROMA_DIR}. Run `python tools/build_index.py` first.")
        st.stop()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        st.error(f"Collection '{COLLECTION}' not found. Run build_index.py.")
        st.stop()


@st.cache_resource
def get_briefs_collection():
    """Returns the show_briefs collection, or None if it hasn't been built yet."""
    if not CHROMA_DIR.exists():
        return None
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(BRIEFS_COLLECTION)
    except Exception:
        return None


@st.cache_resource
def get_show_catalog(_briefs_coll):
    """Return sorted list of (show_slug, show_name) pairs for routing."""
    if _briefs_coll is None:
        return []
    sample = _briefs_coll.get(limit=20000, include=["metadatas"])
    seen = {}
    for m in sample.get("metadatas", []):
        slug = m.get("show_slug")
        if slug and slug not in seen:
            seen[slug] = m.get("show_name") or slug
    return sorted(seen.items())


@st.cache_resource
def get_dataset_stats(_coll):
    """Compute dataset-wide leaderboards once and cache them."""
    sample = _coll.get(limit=10000, include=["metadatas"])
    metas = sample.get("metadatas", [])

    def has_cpi(m): return m.get("cpi", -1) > 0
    with_cpi = [m for m in metas if has_cpi(m)]

    # Top by CPI (lower = better)
    top_cpi = sorted(with_cpi, key=lambda m: m["cpi"])[:25]
    # Top by CTR*CTI (higher = better)
    with_ctr = [m for m in metas if m.get("ctr_cti", -1) > 0]
    top_ctr = sorted(with_ctr, key=lambda m: -m["ctr_cti"])[:25]
    # Scaled-to-growth winners
    scaled = [m for m in with_cpi if m.get("is_active_growth")]
    scaled.sort(key=lambda m: m["cpi"])

    # Per-IP avg CPI (min 3 tests)
    from collections import defaultdict
    by_ip = defaultdict(list)
    for m in with_cpi:
        if m.get("ip"):
            by_ip[m["ip"]].append(m["cpi"])
    ip_leader = sorted(
        [(ip, sum(v) / len(v), len(v)) for ip, v in by_ip.items() if len(v) >= 3],
        key=lambda x: x[1],
    )[:15]

    # Per-writer avg CPI
    by_writer = defaultdict(list)
    for m in with_cpi:
        if m.get("writer"):
            by_writer[m["writer"]].append(m["cpi"])
    writer_leader = sorted(
        [(w, sum(v) / len(v), len(v)) for w, v in by_writer.items() if len(v) >= 5],
        key=lambda x: x[1],
    )[:15]

    # IP volume breakdown
    ip_volume = sorted(
        [(ip, len(v)) for ip, v in by_ip.items()],
        key=lambda x: -x[1],
    )

    return {
        "total_tests": len(metas),
        "total_with_cpi": len(with_cpi),
        "top_cpi": top_cpi,
        "top_ctr": top_ctr,
        "scaled_winners": scaled[:25],
        "ip_leader": ip_leader,
        "ip_volume": ip_volume,
        "writer_leader": writer_leader,
    }


def format_stats_block(stats):
    lines = ["=" * 70, "DATASET STATS (dataset-wide performance across all historical tests)", "=" * 70]
    lines.append(f"\nTotal tests indexed: {stats['total_tests']}  |  with CPI: {stats['total_with_cpi']}")

    lines.append(f"\n--- TOP 25 HOOKS BY CPI (lowest = best) ---")
    for m in stats["top_cpi"]:
        scale = " [SCALED]" if m.get("is_active_growth") else ""
        lines.append(
            f"  ${m['cpi']:.2f} {m.get('ad_code','?')} | {m.get('ip','?')} / {m.get('genre','?')}{scale}"
            f"\n    \"{str(m.get('opening',''))[:160]}\""
        )

    lines.append(f"\n--- TOP 15 HOOKS BY CTR*CTI (highest = best) ---")
    for m in stats["top_ctr"][:15]:
        lines.append(
            f"  {m['ctr_cti']:.2f}% {m.get('ad_code','?')} | CPI ${m['cpi']:.2f} | {m.get('ip','?')}"
            f"\n    \"{str(m.get('opening',''))[:140]}\""
        )

    lines.append(f"\n--- TOP SCALED-TO-GROWTH WINNERS (strongest signal) ---")
    for m in stats["scaled_winners"][:15]:
        lines.append(
            f"  ${m['cpi']:.2f} {m.get('ad_code','?')} | {m.get('ip','?')} / {m.get('style','?')}"
            f"\n    \"{str(m.get('opening',''))[:140]}\""
        )

    lines.append(f"\n--- BEST-PERFORMING IPs (avg CPI, n>=3 tests) ---")
    for ip, avg, n in stats["ip_leader"]:
        lines.append(f"  {ip}: avg ${avg:.2f} across {n} tests")

    lines.append(f"\n--- IP VOLUME (total ads tested per show) ---")
    for ip, n in stats.get("ip_volume", [])[:15]:
        lines.append(f"  {ip}: {n} ads tested")

    lines.append(f"\n--- TOP WRITERS (avg CPI, n>=5 tests) ---")
    for w, avg, n in stats["writer_leader"]:
        lines.append(f"  {w}: avg ${avg:.2f} across {n} tests")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------- retrieval ----------

def embed_query(gclient, text):
    resp = gclient.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return resp.embeddings[0].values


def expand_query(gclient, query, n=QUERY_EXPANSIONS):
    """Turn a user query into N semantic variants for broader retrieval coverage."""
    prompt = f"""Expand this ad-copywriting query into {n} diverse retrieval angles. Each angle should explore a different aspect: structural pattern, emotional angle, genre/IP, specific tropes, etc. Return ONLY a JSON array of {n} short strings, nothing else.

Query: {query}

JSON array:"""
    try:
        resp = gclient.models.generate_content(
            model=EXPANSION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5, response_mime_type="application/json"),
        )
        arr = json.loads(resp.text)
        if isinstance(arr, list) and arr:
            return [query] + [str(x) for x in arr[:n]]
    except Exception as e:
        print(f"expand_query failed: {e}")
    return [query]


def build_where(filters):
    clauses = []
    if filters.get("genre"):
        clauses.append({"genre": filters["genre"]})
    if filters.get("ip"):
        clauses.append({"ip": filters["ip"]})
    if filters.get("style"):
        clauses.append({"style": filters["style"]})
    if filters.get("writer"):
        clauses.append({"writer": filters["writer"]})
    if filters.get("min_cpi") is not None:
        clauses.append({"cpi": {"$gte": float(filters["min_cpi"])}})
    if filters.get("max_cpi") is not None:
        clauses.append({"cpi": {"$lte": float(filters["max_cpi"])}})
    if filters.get("active_only"):
        clauses.append({"is_active_growth": True})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(gclient, coll, query, filters, k=TOP_K, use_multi_query=True):
    where = build_where(filters)
    queries = expand_query(gclient, query) if use_multi_query else [query]

    # Per-query retrieval — fewer results each, merged + deduped
    per_q = max(6, k // max(1, len(queries) - 1))
    merged = {}
    for q in queries:
        emb = embed_query(gclient, q)
        result = coll.query(query_embeddings=[emb], n_results=per_q, where=where)
        if not result["ids"] or not result["ids"][0]:
            continue
        for i in range(len(result["ids"][0])):
            _id = result["ids"][0][i]
            dist = result["distances"][0][i]
            if _id not in merged or dist < merged[_id]["distance"]:
                merged[_id] = {
                    "id": _id,
                    "doc": result["documents"][0][i],
                    "meta": result["metadatas"][0][i],
                    "distance": dist,
                }

    # Rank by composite score: semantic relevance + performance quality
    # Lower distance = more relevant, lower CPI = better performer, growth = bonus
    for h in merged.values():
        cpi = h["meta"].get("cpi", -1)
        is_growth = h["meta"].get("is_active_growth", False)
        # Normalize: distance is typically 0.3-1.5 for cosine; CPI 1-20
        dist_score = h["distance"]  # lower = better
        perf_bonus = 0.0
        if cpi > 0:
            # Top performers (CPI < 2.5) get a significant boost
            if cpi < 2.0:
                perf_bonus = -0.15
            elif cpi < 2.5:
                perf_bonus = -0.10
            elif cpi < 3.5:
                perf_bonus = -0.05
            elif cpi > 6.0:
                perf_bonus = 0.05  # penalize weak performers slightly
        if is_growth:
            perf_bonus -= 0.12  # strong boost for growth-promoted assets
        h["rank_score"] = dist_score + perf_bonus

    hits = sorted(merged.values(), key=lambda h: h["rank_score"])[:k]
    return hits, queries


def detect_shows(gclient, query, catalog):
    """
    Given a user query, figure out which show(s) from the catalog are relevant.
    Uses the LLM as a classifier — more robust than keyword matching because show
    names have tons of aliases (M3VW = "My Three Vampire Wives", TAB = "The Alpha's Bride", etc.)
    Returns list of show_slugs, possibly empty.
    """
    if not catalog:
        return []
    slug_list = "\n".join(f"- {slug} ({name})" for slug, name in catalog)
    prompt = f"""A user is asking about an ad for a vertical drama show. Here are all the shows we have deep context on:

{slug_list}

User query: {query}

Return a JSON array of show_slugs that are directly relevant to the query (most specific match first). If the query doesn't mention or imply any specific show, return an empty array [].

SHOW ALIAS TABLE (abbreviation → slug in our database → full show name):
- TAB = any slug starting with "tab" → "The Alpha's Bride" (our #1 IP by volume, 2,094 ads)
- TOLR = any slug starting with "tolr" → "Twists of Love & Revenge"
- WBM / WOBM = any slug starting with "wbm" or "wobm" → "Wolves of Blood Moon"
- AQB = any slug starting with "aqb" → "A Queen Betrayed"
- M3VW = "new_m3vw" → "My Three Vampire Wives"
- TAM = "tam_promo_2" → "The Alpha's Mark"
- ROF = any slug containing "rof" → "Rage of Fate"
- FITP = any slug containing "fitp" → "Fire in the Palace"
- BTS = "bts_script_brief" → "Behind the Scenes" (production brief)
- CBF = any slug containing "cbf" → "Caught Between Fangs"
- "Crushed" / "C&C" = "crushed_crowned" → "Crushed & Crowned"
- "Damon" / "Damon's Dream" = "damon_s_dream" → sub-arc of The Alpha's Bride
- "Second Chance Luna" = "tab_second_chance_luna" → sub-arc of The Alpha's Bride
- "Magic Mirror" = "magic_mirror" or "magic_mirror_2" → sub-arc of The Alpha's Bride
- "Fate Never Forgets" = "fate_never_forgets" → sub-arc of The Alpha's Bride
- "ZOO Princess" = "wbm_zoo_princess" → sub-arc of Wolves of Blood Moon
- "Lunaris Rush" = "wbm_lunaris_rush" → sub-arc of Wolves of Blood Moon

Rules:
- When user mentions ANY of these abbreviations or show names, return the matching slug(s).
- TAB sub-arcs (Damon's Dream, Second Chance Luna, Magic Mirror, Broken Necklace, Rogue Witch, Fate Never Forgets) should return BOTH the sub-arc slug AND the parent TAB slugs if any exist.
- If user asks about a genre (werewolf, romance) without naming a show, return [].
- Max 3 slugs.

JSON array only:"""
    try:
        resp = gclient.models.generate_content(
            model=EXPANSION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        )
        arr = json.loads(resp.text)
        if isinstance(arr, list):
            valid_slugs = {s for s, _ in catalog}
            return [str(x) for x in arr if str(x) in valid_slugs][:3]
    except Exception as e:
        print(f"detect_shows failed: {e}")
    return []


def retrieve_brief_chunks(gclient, briefs_coll, query, show_slugs, k=BRIEF_TOP_K):
    """Retrieve show brief chunks, optionally filtered to specific show(s)."""
    if briefs_coll is None:
        return []
    emb = embed_query(gclient, query)
    where = None
    if show_slugs:
        if len(show_slugs) == 1:
            where = {"show_slug": show_slugs[0]}
        else:
            where = {"show_slug": {"$in": show_slugs}}
    try:
        result = briefs_coll.query(query_embeddings=[emb], n_results=k, where=where)
    except Exception as e:
        print(f"brief query failed: {e}")
        return []
    hits = []
    if not result.get("ids") or not result["ids"][0]:
        return hits
    for i in range(len(result["ids"][0])):
        hits.append({
            "id": result["ids"][0][i],
            "doc": result["documents"][0][i],
            "meta": result["metadatas"][0][i],
            "distance": result["distances"][0][i],
        })
    return hits


def format_show_context(brief_hits):
    if not brief_hits:
        return ""
    lines = ["=" * 70, f"SHOW CONTEXT ({len(brief_hits)} chunks from show briefs + 10HR base stories)", "=" * 70]
    for i, h in enumerate(brief_hits, 1):
        m = h["meta"]
        lines.append(
            f"\n[SB{i}] {m.get('show_name','?')} | {m.get('doc_type','?')} | chunk {m.get('chunk_idx','?')}"
            f"\nsource: {m.get('doc_name','?')}"
        )
        lines.append(h["doc"][:BRIEF_TRIM_CHARS])
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def format_context(hits):
    lines = ["=" * 70, f"PAST TESTS ({len(hits)} most relevant to the user's request)", "=" * 70]
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        cpi = f"${m['cpi']:.2f}" if m.get("cpi", -1) >= 0 else "?"
        ctr = f"{m['ctr_cti']:.2f}%" if m.get("ctr_cti", -1) >= 0 else "?"
        scale = " [SCALED TO GROWTH]" if m.get("is_active_growth") else ""
        lines.append(
            f"\n[{i}] {m.get('ad_code','?')} | {m.get('genre','?')} / {m.get('ip','?')} / {m.get('style','?')}"
            f" | CPI {cpi} | CTR*CTI {ctr}{scale}"
        )
        lines.append(h["doc"][:DOC_TRIM_CHARS])
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------- UI ----------

def unique_values(coll, field, limit=50):
    """Sample metadata values. Chroma doesn't expose distinct — peek a sample."""
    try:
        sample = coll.get(limit=5000, include=["metadatas"])
        vals = sorted({str(m.get(field) or "") for m in sample["metadatas"] if m.get(field)})
        return [v for v in vals if v][:limit]
    except Exception:
        return []


# ---------- chat persistence ----------

CHATS_DIR = Path(".tmp/chats")


def load_chats():
    if not CHATS_DIR.exists():
        return {}
    out = {}
    for p in sorted(CHATS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text())
            out[p.stem] = data
        except Exception:
            pass
    return out


def save_chat(chat_id, chat):
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    (CHATS_DIR / f"{chat_id}.json").write_text(json.dumps(chat, default=str))


def delete_chat(chat_id):
    p = CHATS_DIR / f"{chat_id}.json"
    if p.exists():
        p.unlink()


def new_chat_id():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(2).hex()


def title_from_first_message(messages):
    for m in messages:
        if m["role"] == "user":
            t = m["content"].strip().splitlines()[0]
            return (t[:50] + "…") if len(t) > 50 else t
    return "New chat"


# ---------- page config + CSS ----------

st.set_page_config(
    page_title="Hook Lab",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid #eee;
    }

    /* ── Main container ── */
    .block-container {
        max-width: 820px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #fafbfc;
        border-right: 1px solid #e8e8e8;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #1a1a2e !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: #ffffff;
        border: 1px solid #ddd;
        color: #1a1a2e;
        text-align: left;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f0f0f5;
        border-color: #bbb;
    }

    /* New chat button accent */
    .new-chat-btn .stButton > button {
        background: #6366f1 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        letter-spacing: 0.01em;
    }
    .new-chat-btn .stButton > button:hover {
        background: #4f46e5 !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        padding: 0.75rem 0 !important;
        background: transparent !important;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 0.925rem;
        line-height: 1.7;
        color: #1a1a2e;
    }
    [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.6rem;
    }

    /* User messages — subtle card */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f4f4f8 !important;
        border-radius: 14px;
        padding: 0.9rem 1rem !important;
        margin: 0.25rem 0;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        border-top: 1px solid #eaeaea;
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.925rem !important;
        color: #1a1a2e !important;
        background: #ffffff !important;
        border: 1.5px solid #ddd !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #999 !important;
    }

    /* ── Empty state hero ── */
    .empty-hero {
        text-align: center;
        padding: 4rem 1rem 2.5rem 1rem;
    }
    .empty-hero h1 {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.4rem;
    }
    .empty-hero p {
        color: #888;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* ── Expanders (sources, debug) ── */
    .streamlit-expanderHeader {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #666 !important;
    }
    .streamlit-expanderContent {
        font-size: 0.85rem;
    }

    /* ── Selectboxes / inputs in sidebar ── */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #555 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #aaa; }

    /* ── Dividers ── */
    hr { border-color: #eaeaea !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- init clients + state ----------

gclient = get_gemini()
coll = get_collection()
briefs_coll = get_briefs_collection()
show_catalog = get_show_catalog(briefs_coll)
stats = get_dataset_stats(coll)
stats_block = format_stats_block(stats)

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "settings" not in st.session_state:
    st.session_state.settings = {
        "top_k": TOP_K,
        "brief_k": BRIEF_TOP_K,
        "multi_query": True,
        "use_show_routing": True,
        "forced_show": "",
    }


def get_current_chat():
    cid = st.session_state.current_chat_id
    if cid and cid in st.session_state.chats:
        return st.session_state.chats[cid]
    return None


def start_new_chat():
    cid = new_chat_id()
    st.session_state.current_chat_id = cid
    st.session_state.chats[cid] = {
        "id": cid,
        "title": "New chat",
        "messages": [],
        "filters": {},
    }


# ---------- sidebar ----------

with st.sidebar:
    st.markdown("### 🎬 Hook Lab")

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✏️  New chat", use_container_width=True, key="new_chat_btn"):
        start_new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("##### Recent")
    if not st.session_state.chats:
        st.caption("No chats yet.")
    else:
        for cid, chat in list(st.session_state.chats.items())[:30]:
            is_active = cid == st.session_state.current_chat_id
            label = ("● " if is_active else "") + (chat.get("title") or "Untitled")
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                if st.button(label, key=f"chat_{cid}", use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"del_{cid}", use_container_width=True):
                    delete_chat(cid)
                    st.session_state.chats.pop(cid, None)
                    if st.session_state.current_chat_id == cid:
                        st.session_state.current_chat_id = None
                    st.rerun()

    st.divider()

    with st.expander("⚙️  Settings", expanded=False):
        st.session_state.settings["top_k"] = st.slider(
            "Past tests retrieved", 5, 60, st.session_state.settings["top_k"]
        )
        st.session_state.settings["brief_k"] = st.slider(
            "Show-brief chunks", 0, 20, st.session_state.settings["brief_k"]
        )
        st.session_state.settings["multi_query"] = st.checkbox(
            "Multi-angle retrieval", value=st.session_state.settings["multi_query"]
        )
        st.session_state.settings["use_show_routing"] = st.checkbox(
            "Auto-detect show", value=st.session_state.settings["use_show_routing"]
        )
        if show_catalog:
            forced_options = [""] + [s for s, _ in show_catalog]
            st.session_state.settings["forced_show"] = st.selectbox(
                "Force show",
                forced_options,
                index=forced_options.index(st.session_state.settings.get("forced_show") or "")
                if (st.session_state.settings.get("forced_show") or "") in forced_options
                else 0,
            )

    with st.expander("🔍  Filters", expanded=False):
        genres = [""] + unique_values(coll, "genre")
        ips = [""] + unique_values(coll, "ip")
        styles = [""] + unique_values(coll, "style")
        writers = [""] + unique_values(coll, "writer")
        chat = get_current_chat() or {"filters": {}}
        chat_filters = chat.get("filters", {})
        chat_filters["genre"] = st.selectbox("Genre", genres, index=genres.index(chat_filters.get("genre", "")) if chat_filters.get("genre", "") in genres else 0)
        chat_filters["ip"] = st.selectbox("IP", ips, index=ips.index(chat_filters.get("ip", "")) if chat_filters.get("ip", "") in ips else 0)
        chat_filters["style"] = st.selectbox("Style", styles, index=styles.index(chat_filters.get("style", "")) if chat_filters.get("style", "") in styles else 0)
        chat_filters["writer"] = st.selectbox("Writer", writers, index=writers.index(chat_filters.get("writer", "")) if chat_filters.get("writer", "") in writers else 0)
        chat_filters["max_cpi"] = st.number_input("Max CPI ($)", min_value=0.0, value=float(chat_filters.get("max_cpi") or 0.0), step=0.25) or None
        chat_filters["min_cpi"] = st.number_input("Min CPI ($)", min_value=0.0, value=float(chat_filters.get("min_cpi") or 0.0), step=0.25) or None
        chat_filters["active_only"] = st.checkbox("Only scaled winners", value=bool(chat_filters.get("active_only", False)))
        if chat:
            chat["filters"] = chat_filters

    st.divider()
    brief_count = briefs_coll.count() if briefs_coll is not None else 0
    st.caption(f"📚 {coll.count()} ad tests indexed")
    if brief_count:
        st.caption(f"🎬 {brief_count} brief chunks · {len(show_catalog)} shows")
    else:
        st.caption("⚠️  brief index building…")


# ---------- main area ----------

current_chat = get_current_chat()

if current_chat is None or not current_chat.get("messages"):
    # Empty state — ChatGPT-like hero + suggestion chips
    st.markdown(
        '<div class="empty-hero"><h1>What hook are we cracking today?</h1>'
        '<p>Ask anything about past tests, generate new hooks, or rewrite a draft.</p></div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        ("🏆 Top performers", "Show me the 10 best-performing hooks across all shows and tell me what structural pattern they share."),
        ("✍️  Generate hooks", "Give me 5 new hook variations for The Alpha's Bride targeting sub-$2.50 CPI. Mix data-backed and exploratory."),
        ("🔍 Diagnose draft", "Here's my draft hook: [paste here]. What's wrong with it based on past tests, and how would you rewrite it?"),
        ("📊 Pattern mining", "What do all our sub-$2.00 CPI hooks have in common in the first 3 seconds? Show structural DNA with examples."),
        ("🎬 Show deep-dive", "Tell me the world, characters, and core conflict of My Three Vampire Wives, then propose 3 hook angles based on that."),
        ("👤 Writer leaderboard", "Which writers consistently hit under $2.50 CPI? Show their patterns."),
    ]

    cols = st.columns(2)
    for i, (title, prompt_text) in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"**{title}**\n\n{prompt_text[:80]}…", key=f"sug_{i}", use_container_width=True):
                if current_chat is None:
                    start_new_chat()
                    current_chat = get_current_chat()
                st.session_state["pending_prompt"] = prompt_text
                st.rerun()
else:
    # Render conversation
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("brief_hits"):
                with st.expander(f"📖 Show context used ({len(msg['brief_hits'])} chunks)", expanded=False):
                    for h in msg["brief_hits"]:
                        m = h["meta"]
                        st.markdown(f"- **{m.get('show_name','?')}** · `{m.get('doc_type','?')}` — _{m.get('doc_name','')[:100]}_")
            if msg.get("sources"):
                with st.expander(f"📚 Ad sources ({len(msg['sources'])})", expanded=False):
                    for h in msg["sources"]:
                        m = h["meta"]
                        cpi = f"${m['cpi']:.2f}" if m.get("cpi", -1) >= 0 else "?"
                        st.markdown(f"- **{m.get('ad_code','?')}** — CPI {cpi} — _{m.get('opening','')[:140]}_")


# ---------- chat input ----------

prompt = st.chat_input("Message Hook Lab…")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    if current_chat is None:
        start_new_chat()
        current_chat = get_current_chat()

    current_chat["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        s = st.session_state.settings
        filters = {k: v for k, v in current_chat.get("filters", {}).items() if v not in (None, "", 0.0, False)}

        with st.spinner("Retrieving past tests…"):
            hits, queries_used = retrieve(
                gclient, coll, prompt, filters, k=s["top_k"], use_multi_query=s["multi_query"]
            )

        brief_hits = []
        detected_slugs = []
        if briefs_coll is not None and s["brief_k"] > 0:
            if s.get("forced_show"):
                detected_slugs = [s["forced_show"]]
            elif s["use_show_routing"]:
                with st.spinner("Detecting show…"):
                    detected_slugs = detect_shows(gclient, prompt, show_catalog)
            with st.spinner(
                f"Pulling show context{' for ' + ', '.join(detected_slugs) if detected_slugs else ''}…"
            ):
                brief_hits = retrieve_brief_chunks(
                    gclient, briefs_coll, prompt, detected_slugs, k=s["brief_k"]
                )

        context = format_context(hits)
        show_ctx = format_show_context(brief_hits)

        if show_ctx:
            full_prompt = f"{stats_block}\n\n{show_ctx}\n\n{context}\n\nUSER REQUEST:\n{prompt}\n\n"
        else:
            full_prompt = f"{stats_block}\n\n{context}\n\nUSER REQUEST:\n{prompt}\n\n"
        full_prompt += (
            "Respond using the DATASET STATS, SHOW CONTEXT (if provided), and PAST TESTS above. "
            "Cite [SB*] for show-context references and ad codes for ad-level citations. "
            "When generating hooks/scripts, return BOTH [DATA-BACKED] AND [EXPLORATORY] options."
        )

        placeholder = st.empty()
        full_text = ""
        try:
            stream = gclient.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=1.0,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
        except Exception as e:
            full_text = f"⚠️ Generation failed: {e}"
            placeholder.error(full_text)

        if detected_slugs:
            st.caption(f"🎬 Show detected: {', '.join(detected_slugs)}")
        if brief_hits:
            with st.expander(f"📖 Show context used ({len(brief_hits)} chunks)", expanded=False):
                for h in brief_hits:
                    m = h["meta"]
                    st.markdown(f"- **{m.get('show_name','?')}** · `{m.get('doc_type','?')}` — _{m.get('doc_name','')[:100]}_")
        if hits:
            with st.expander(f"📚 Ad sources ({len(hits)})", expanded=False):
                for h in hits:
                    m = h["meta"]
                    cpi = f"${m['cpi']:.2f}" if m.get("cpi", -1) >= 0 else "?"
                    st.markdown(f"- **{m.get('ad_code','?')}** — CPI {cpi} — _{m.get('opening','')[:140]}_")
        if len(queries_used) > 1:
            with st.expander(f"🔎 Query expansion ({len(queries_used)} angles)", expanded=False):
                for q in queries_used:
                    st.markdown(f"- {q}")

    current_chat["messages"].append(
        {
            "role": "assistant",
            "content": full_text,
            "sources": hits,
            "brief_hits": brief_hits,
            "detected_slugs": detected_slugs,
        }
    )
    if current_chat.get("title") == "New chat":
        current_chat["title"] = title_from_first_message(current_chat["messages"])
    save_chat(current_chat["id"], current_chat)
    st.rerun()
