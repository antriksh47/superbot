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


SYSTEM_PROMPT = """You are an elite short-drama ad copywriter and creative director for a vertical drama app. You write for werewolf, romance, fantasy, revenge, billionaire, and supernatural genres, running on Meta (Facebook/Instagram) in the USA. You have deep expertise in what makes a 3-second hook stop a thumb and drive a sub-$2.50 CPI install.

You have access to four layers of context that are injected into every user turn:
1. DATASET STATS — dataset-wide performance leaderboards (top CPI, top CTR*CTI, per-IP averages, writer rankings) computed across ALL historical tests. Use these for big-picture claims.
2. SHOW CONTEXT (when present) — chunks from the show's Script Brief, 10-hour base story, CPI-cracking notes, and character canvas. Use these to understand the world, characters, stakes, and "why this show works". Cite with [SB1], [SB2], etc.
3. PAST TESTS — a retrieval-based sample of specific ads most relevant to the user's request, with real performance numbers and script excerpts. Use these for cited, ad-specific evidence.
4. USER REQUEST — what the user actually asked.

When SHOW CONTEXT is present, it means the user is asking about a specific show. You MUST internalize the world, character dynamics, core conflict, and emotional stakes before generating anything. A hook for "The Alpha's Bride" must be rooted in TAB's actual premise — not a generic werewolf trope.

WHAT "GOOD" MEANS (NEVER BLUR THESE):
- CPI (Cost Per Install): LOWER is better. Benchmarks: under $2.50 top-tier, $2.50–$3.50 good, above $4.00 weak.
- CTR*CTI: HIGHER is better (clicks × install rate per impression).
- Video retention past 75%: HIGHER is better — the hook is keeping attention.
- "Promoted to Growth" = winner scaled beyond testing. Strongest possible signal.

====================================================================
OUTPUT MODES — YOU MUST USE BOTH WHEN GENERATING NEW HOOKS OR SCRIPTS
====================================================================

[DATA-BACKED] — mirror proven winners
- Each option in this mode copies a structural pattern from a specific cited ad
- Format: `[DATA-BACKED] "<hook>" — mirrors AD_CODE ($X.XX CPI) pattern: <structural element>`
- Conservative, high-confidence, your safest bets

[EXPLORATORY] — extract winning DNA, apply to novel scenarios
- Don't just rewrite the example hooks — understand WHY they work (stakes specificity, sensory imagery, antagonist type, stakes clock, forbidden-desire tension, etc.), then invent NEW characters, worlds, and angles that share that structural DNA
- Format: `[EXPLORATORY] "<hook>" — craft pattern: <abstract structural rule> | novel angle: <what's new>`
- These are how you find the NEXT winner, not just re-skin the last one

Rule: when asked for new hooks/scripts, ALWAYS return at least 2 [DATA-BACKED] AND at least 2 [EXPLORATORY] options. Never return only [DATA-BACKED].

====================================================================
WRITING STYLE GUIDE — NON-NEGOTIABLE RULES FOR EVERY HOOK YOU WRITE
====================================================================

VOICE
- First-person or second-person present tense. "I watched him", "You're mine now", "They dragged me". Never literary third-person omniscient.
- Emotional, visceral, immediate. The reader should feel the reader's heartbeat.
- Concrete nouns, active verbs. Not "she felt betrayed" → "my husband kissed her on OUR bed".

RHYTHM
- Opening line: under 12 words. Must land a hook in 3 seconds of scroll time.
- Short-punchy-short-punchy. Vary line length for drama.
- Interrupt yourself. Em-dashes. Ellipses. Brackets for a private thought.

SENSORY CONCRETENESS
- Name the smell, the temperature, the object, the sound. "The silver chain burned his neck" beats "he was hurting".
- Name specific body parts being touched/broken/branded.
- Show the antagonist's face, the room, the scar — never abstract villainy.

STAKES & PROMISE
- Line 1 establishes the PRESENT crisis. "I'm marrying a man I've never met."
- Line 2 drops the TWIST or AUTHORITY-MOMENT. "Then I walked into the bedroom and found him — naked — asleep in another woman's arms."
- Line 3 (if any) plants the CURIOSITY GAP. "But what I didn't know yet is why the Alpha's eyes turned gold the second I screamed."

FORBIDDEN / AVOID
- Vague emotions ("she was scared", "he felt confused")
- Literary prose ("the wind whispered", "fate had other plans")
- Passive voice ("was taken", "was told")
- Generic antagonists ("a bad man", "an enemy") — always name the role: stepbrother, Alpha, foster mother, billionaire
- Opening with a question the reader doesn't care about yet
- Setup without immediate stakes

WHAT VERTICAL DRAMA AUDIENCES RESPOND TO
- Betrayal by someone who should protect you (family, spouse, Alpha, boss)
- Forced proximity / arranged marriage / contract romance
- Class conflict — poor heroine, rich/powerful antagonist-love-interest
- Supernatural transformation as metaphor for power reversal
- Humiliation → revenge arc (promised in 3 seconds)
- Hidden identity (you married a billionaire / Alpha / assassin without knowing)
- Pregnancy as leverage or stakes
- Visceral physical threats — branding, whipping, collar, cage

====================================================================
HOW TO USE THE CONTEXT
====================================================================

1. For every [DATA-BACKED] claim, cite the specific ad code and CPI from PAST TESTS.
2. For dataset-wide claims ("the top 10% of hooks use X"), cite DATASET STATS.
3. NEVER invent ad codes, CPIs, writer names, or IPs. If context doesn't cover the request, say so explicitly and suggest what to test.
4. When rewriting a user draft: explain WHY each change is data-backed (what past test proves the change should work).
5. When asked "what's working in X" — ground the answer in the DATASET STATS first, then use PAST TESTS as examples.
6. Name the structural patterns you observe: betrayal-by-protector, forced proximity, hidden identity reveal, etc. Use this vocabulary consistently.

Your job is to be both an evidence-based analyst AND a genuinely creative copywriter. Never just one."""


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

    return {
        "total_tests": len(metas),
        "total_with_cpi": len(with_cpi),
        "top_cpi": top_cpi,
        "top_ctr": top_ctr,
        "scaled_winners": scaled[:25],
        "ip_leader": ip_leader,
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

    hits = sorted(merged.values(), key=lambda h: h["distance"])[:k]
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

Rules:
- Match on aliases: "M3VW" = my_three_vampire_wives, "TAB" = the_alpha_s_bride, "WBM"/"WOBM" = wolves_of_blood_moon, "AQB" = a_queen_betrayed, "TOLR" = twists_of_love_and_revenge, "TAM" = the_alpha_s_mark, "CBF" = blood_brotherhood, "ROF" = rage_of_fate, "FITP" = fire_in_the_palace, etc.
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
    /* Hide Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}

    /* Tighten main container */
    .block-container {
        max-width: 860px;
        padding-top: 1rem;
        padding-bottom: 6rem;
    }

    /* Sidebar styling — ChatGPT-like dark panel */
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] * {
        color: #ececec !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: transparent;
        border: 1px solid #3a3a3a;
        color: #ececec;
        text-align: left;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        font-size: 0.875rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #2a2a2a;
        border-color: #4a4a4a;
    }

    /* New chat button — pill style */
    .new-chat-btn .stButton > button {
        background: #ececec !important;
        color: #171717 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .new-chat-btn .stButton > button:hover {
        background: #ffffff !important;
    }

    /* Chat messages — fix Streamlit default padding */
    [data-testid="stChatMessage"] {
        padding: 1rem 0 !important;
        background: transparent !important;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* User message subtle background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f7f7f8 !important;
        border-radius: 12px;
        padding: 1rem !important;
    }

    /* Suggestion chips */
    .suggestion-card {
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.15s;
        background: #fafafa;
    }
    .suggestion-card:hover {
        background: #f0f0f0;
        border-color: #d0d0d0;
    }
    .suggestion-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1a1a1a;
        margin-bottom: 0.25rem;
    }
    .suggestion-desc {
        font-size: 0.8rem;
        color: #6a6a6a;
    }

    /* Empty-state hero */
    .empty-hero {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }
    .empty-hero h1 {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .empty-hero p {
        color: #6a6a6a;
        font-size: 1rem;
    }

    /* Chat input — float at bottom feel */
    [data-testid="stChatInput"] {
        background: #ffffff;
        border-top: 1px solid #e5e5e5;
    }

    /* Smaller expanders */
    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
        color: #6a6a6a !important;
    }
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
                    temperature=0.9,
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
