"""
Streamlit chatbot: data-backed ad script generator.
See workflows/04_run_chatbot.md for full spec.

Run:
    streamlit run tools/chatbot_app.py
"""
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Import structured data tools — these read JSON directly, no embeddings needed
from data_tools import (
    query_assets,
    get_asset_detail,
    get_show_context,
    get_opening_stats,
    get_writer_stats,
    get_leaderboard,
    _load_assets,
    _load_briefs,
    SHOW_ALIASES,
)

CHAT_MODEL = "gemini-2.5-flash"


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
SHOW CHARACTER BIBLES — USE THESE EXACT NAMES AND DETAILS
====================================================================
NEVER hallucinate character names. Use ONLY these characters. If you need more detail, call get_show_context().

--- THE ALPHA'S BRIDE (TAB) — Werewolf Romance ---
PROTAGONIST: Talia — 19yo wolfless orphan living in the attic of the Red Moon packhouse. Brought as a toddler by the previous Alpha (who died). Her wolf spoke once 4 years ago: "You're too weak to shift. If you force it, it might kill you." No speed, strength, healing. Malnourished, wears discarded clothes, sleeps on a torn futon. Works as invisible Omega cleaning bathrooms at night. No mind link (never had pack-joining ceremony). Bullied by Anna (ringleader), Shawn, Zina.

LOVE INTEREST: Damon — Alpha of the Dark Howlers pack (largest in North America). 27yo, raven black hair, blue eyes. Became Alpha at 17 after parents died in rogue ambush. Can verbally talk to his wolf (unique ability). Serial one-night-stand man, rejects every bride candidate. His wolf identifies Talia as his fated mate when she sneaks into his guest room — catches her "sweet citrusy scent of freesia."

ANTAGONISTS:
- Alpha Edward — Alpha of Red Moon pack (2nd largest). Marcy's father. Power-hungry, treats Omegas as invisible.
- Luna Layla — Edward's wife. Cares only about image. Complicit in using Marcy as political pawn.
- Marcy Redmayne — Edward and Layla's daughter. Blonde, blue/grey eyes. Spent 10 years in Europe. Found her true fated mate George (an Omega) at her welcome party but REJECTED him because of his low status. Accepted arranged marriage to Damon. Performed oral sex on Damon in his guest room — this is when Talia walked in.
- Anna — Omega bully ringleader who deliberately sent Talia to Damon's room as sabotage.
- Elder Parker — pushes Damon to marry Marcy for political gain.

SUPPORTING: Caden (Damon's Beta, mated to Maya), Stephanie (Dark Howlers caretaker, was friends with Damon's late mother Luna Violet), Nora (Beta Raymond's daughter, Marcy's handler), James (Marcy's 15yo brother), George (Marcy's rejected true mate, Omega), Olivia (Talia's only friend, now gone).

PACKS: Dark Howlers (Damon's, largest), Red Moon (Edward's, 2nd largest), Steelbite (smaller, Cassie's father's).

WORLD RULES: Fated mates sensed after 18th birthday via scent + instant bond. Bond can be rejected (causes immense pain). "Wolfless" = wolf dormant, no werewolf abilities. Mind link = pack telepathy, requires joining ceremony. Omegas = lowest rank.

CORE CONFLICT: Damon arrives at Red Moon to reject arranged bride Marcy — discovers his true fated mate is Talia, an invisible wolfless orphan in the attic. Tension: Damon's emotional walls vs Talia's total powerlessness vs political machinery forcing the Marcy marriage vs danger from bullies.

KEY OPENING BEATS: Talia bullied → Anna sends her to Damon's room with towels → she walks in on Marcy with Damon → flees → Damon's wolf declares "MATE" → he smells freesia → pushes Marcy away → hunts for the mystery girl.

--- TWISTS OF LOVE & REVENGE (TOLR) — Billionaire Revenge Romance ---
NOT a werewolf show. Modern-day corporate California.

PROTAGONIST: Susan Drew — 22yo, youngest daughter of Noah Drew and Luna Drew. Architecture/fashion design graduate. Just closed the Aton Architectural deal in Paris. Returns to find her boyfriend stolen, her inheritance given to her sister, and herself being married off as a political pawn.

LOVE INTEREST: Ethan Williams — uses "Storm" publicly (mother's family name). Actually heir of the ultra-wealthy Williams family. CEO of Omini Corporation. Cold, blunt, handsome. Assistant: Felix Knight (drives a Bentley, calls him "Young Master Williams"). Came to California to fulfill his dead mother's promise to the Drew family. Agrees to a "contract marriage" with Susan — "a few years, then we part ways, no children."

ANTAGONISTS:
- Sophia Drew — Susan's elder sister. Stole Jacob from Susan. Named heir (80% of Drew Corp shares). Manipulative, plays innocent.
- Jacob Smith — Susan's childhood sweetheart. Dumped her for Sophia because Sophia inherited the fortune. Tries to convince Susan to "wait a few years."
- Noah Drew — Susan's father. Cold, calculating. Named Sophia heir. Plans to use Susan as "scapegoat" to fulfill promise to Storm family.
- Luna Drew — Susan's mother. Cruel (slapped Susan). Views Storm family as beneath them.

SUPPORTING: Helen (Susan's best friend, loyal, humorous), Felix Knight (Ethan's assistant/butler).

CORE CONFLICT: Susan, betrayed by sister (stole boyfriend + inheritance) and discarded by parents (marrying her off), impulsively marries Ethan — a man she thinks is "Jacob's cousin" but who is secretly a hidden billionaire with his own agenda. Neither knows the other's true motivations.

KEY OPENING BEATS: Susan returns from Paris → finds Jacob and Sophia intimate → slapped by mother → Sophia named heir at party → Susan spots Ethan, kisses him → proposes marriage → next morning they marry at Recorder's Office → contract marriage established → she moves into his deliberately modest villa at Opal Harbor.

--- WOLVES OF BLOOD MOON (WBM) — Supernatural Academy ---

PROTAGONIST: Violet Purple — human girl, purple hair, from a poor background. Mother Nancy is implied sex worker who wanted Violet to follow. Accepted to Lunaris Academy on scholarship despite deliberately offensive application. Sarcastic, tough, defiant. She is HUMAN (not a werewolf) in a school full of werewolf royalty.

THE TERROR FOUR / CARDINAL ALPHAS (born simultaneously during celestial event, each has unique power):
- Asher Nightshade — Alpha of West House. Wears dark shades indoors. Power: MIND-BENDING (plant thoughts, erase memories, puppet people). Orchestrated Violet's admission. Pulled her into a manipulated dream, demanded "Be my queen, rule with me." Cut a lock of her hair IN the dream — it was actually missing when she woke. Calls her "my purple flower."
- Alaric — Alpha of North House. Power: CONTROLS LIGHTNING (EMP bolts, paralysis, storms). Shared an electric spark when Violet touched him in the infirmary.
- Griffin Hale — Alpha of East House. Red-haired. Power: SUPERHUMAN STRENGTH (shockwave punches). CHOKED Violet on first day after she was found holding his stolen necklace. The "brute."
- Roman Draven — Alpha of South House. Green hair. Power: SHAPESHIFTING (hawk, lion, wolf). Crashed into Violet day one, gave her Griffin's stolen necklace, fled. A flirt. Hates Griffin.

ANTAGONISTS: Asher (obsessive), Griffin (violent), Hazel Sterling (hostile roommate), Nancy (Violet's neglectful mother).

SUPPORTING: Alice Parker (friendly roommate, gossipy), Lily Fairchild (measured roommate), Catherine (student guide), Principal Jameson (afraid of Cardinal Alphas).

WORLD RULES: Lunaris Academy has 4 Houses (West/East/South/North). Lunaboard = girl ranking system (top 20 may associate with Alphas). Biodome = school social network. The Terror Four were created by Moon Goddess as revenge after humans killed ~90% of she-wolves. Violet shot to Lunaboard #20 on day one.

CORE CONFLICT: Violet, a powerless human, is trapped at an elite werewolf academy where 4 supernaturally gifted Alpha wolves rule. Asher specifically engineered her admission and is obsessed with making her his queen. She must survive violent hierarchy, dream manipulation, and werewolf politics while refusing to submit — completely powerless in a school where power is everything.

KEY OPENING BEATS: Violet arrives → Roman crashes into her, leaves her with Griffin's stolen necklace → Griffin chokes her → Asher appears: "Welcome, my purple flower" → roommates explain Terror Four + Lunaboard → Asher pulls her into dream, demands she be his queen → she wakes to find hair actually cut.

====================================================================
TERMINOLOGY — LEARN THIS BEFORE ANYTHING ELSE
====================================================================

- HOOK / OPENING = The first 1-2 sentences (3 seconds of video). This is what stops the thumb-scroll. Always under 15 words.
- Q1 / OPENING SCRIPT = The full 2-minute opening narration (~450-530 words). This is NOT just the hook — it is a self-contained mini-pilot that must work as a standalone ad while being faithful to the show's world and characters.
- FULL SCRIPT = The entire 8-12 minute ad script (~5,500-6,000 words). Includes Q1 through Q4 + CTA.
- OPENING CODE = Identifier for a specific opening. One proven opening can be reused across many ad codes. If an opening_code has been used in 20+ ads, it is a PROVEN WINNER — prioritize it.
- PROMOTED TO GROWTH / SCALED = This asset performed so well in testing that it was moved to the scaling budget. This is the strongest possible performance signal.

====================================================================
YOUR TOOLS — USE THEM AGGRESSIVELY
====================================================================

You have 6 data tools you MUST call before generating any content. NEVER guess or rely on memory alone — always call the relevant tool first.

1. query_assets(ip, genre, writer, max_cpi, growth_only, sort_by, limit, search_text)
   → Exact filtered search across 2,400+ ad assets. Use for "show me TAB ads under $2 CPI" etc.

2. get_asset_detail(ad_code)
   → Full record for one ad: all metrics + full opening text + full script text.

3. get_show_context(show, section, max_chars)
   → 10-hour base story, character canvas, CPI-cracking notes for a show.
   → CALL THIS before writing any Q1 script — you need real characters and world.

4. get_opening_stats(opening_code, top_n_reused)
   → Stats for a specific opening (reuse count, avg CPI) or the top most-reused openings.
   → Reuse count is a STRONG signal — an opening used in 40+ ads is a proven winner.

5. get_writer_stats(writer, top_n)
   → Writer portfolio or leaderboard. Use for "who writes the best werewolf hooks?"

6. get_leaderboard(metric, ip, genre, growth_only, limit)
   → Top N assets by any metric (CPI, CTR, retention, spend).

TOOL USAGE RULES:
- When asked to write a Q1 or opening for a SPECIFIC show: call get_show_context + query_assets for that show's top performers. You need BOTH the story world AND the proven ad patterns.
- When asked "what's working" or "what are the best hooks": call get_leaderboard or query_assets with appropriate filters. Don't guess.
- When a user mentions a specific ad code: call get_asset_detail to get the full record.
- When asked about a writer: call get_writer_stats.
- When asked about reuse or proven openings: call get_opening_stats.
- You can call MULTIPLE tools in one turn. Do it.

DATASET STATS are also injected into every turn as a baseline leaderboard.

When writing for a specific show, ALWAYS call get_show_context first. You MUST internalize the world, character names, relationships, and core conflict before writing. A Q1 for The Alpha's Bride must use Aria, Alpha Damon, Marcy, the Red Moon Pack — not generic "the girl" and "the Alpha."

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


@st.cache_data
def get_show_catalog():
    """Return sorted list of (show_slug, show_name) pairs from briefs JSON."""
    briefs = _load_briefs()
    seen = {}
    for b in briefs:
        slug = b.get("show_slug")
        name = b.get("name") or b.get("anchor_text") or slug
        if slug and slug not in seen:
            seen[slug] = name
    return sorted(seen.items())


@st.cache_data
def get_dataset_stats():
    """Compute dataset-wide leaderboards from assets JSON. No Chroma needed."""
    from collections import defaultdict
    assets = _load_assets()

    with_cpi = [a for a in assets if a.get("cpi") is not None and a["cpi"] > 0]

    # Top by CPI (lower = better)
    top_cpi = sorted(with_cpi, key=lambda a: a["cpi"])[:25]
    # Top by CTR*CTI (higher = better)
    with_ctr = [a for a in assets if a.get("ctr_cti") is not None and a["ctr_cti"] > 0]
    top_ctr = sorted(with_ctr, key=lambda a: -a["ctr_cti"])[:25]
    # Scaled-to-growth winners
    scaled = [a for a in with_cpi if a.get("is_active_growth")]
    scaled.sort(key=lambda a: a["cpi"])

    # Per-IP avg CPI (min 3 tests)
    by_ip = defaultdict(list)
    for a in with_cpi:
        if a.get("ip"):
            by_ip[a["ip"]].append(a["cpi"])
    ip_leader = sorted(
        [(ip, sum(v) / len(v), len(v)) for ip, v in by_ip.items() if len(v) >= 3],
        key=lambda x: x[1],
    )[:15]

    # Per-writer avg CPI
    by_writer = defaultdict(list)
    for a in with_cpi:
        if a.get("writer"):
            by_writer[a["writer"]].append(a["cpi"])
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
        "total_tests": len(assets),
        "total_with_cpi": len(with_cpi),
        "top_cpi": [{k: a.get(k) for k in ("ad_code", "cpi", "ip", "genre", "opening", "is_active_growth", "ctr_cti")} for a in top_cpi],
        "top_ctr": [{k: a.get(k) for k in ("ad_code", "cpi", "ip", "genre", "opening", "ctr_cti")} for a in top_ctr],
        "scaled_winners": [{k: a.get(k) for k in ("ad_code", "cpi", "ip", "style", "opening", "is_active_growth")} for a in scaled[:25]],
        "ip_leader": ip_leader,
        "ip_volume": ip_volume,
        "writer_leader": writer_leader,
    }


def unique_ip_values():
    """Get unique IP/genre/style/writer values from assets JSON for sidebar dropdowns."""
    assets = _load_assets()
    result = {}
    for field in ("genre", "ip", "style", "writer"):
        vals = sorted({str(a.get(field) or "") for a in assets if a.get(field)})
        result[field] = [v for v in vals if v][:50]
    return result


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


# ---------- Gemini function calling declarations ----------

TOOL_FUNCTIONS = {
    "query_assets": query_assets,
    "get_asset_detail": get_asset_detail,
    "get_show_context": get_show_context,
    "get_opening_stats": get_opening_stats,
    "get_writer_stats": get_writer_stats,
    "get_leaderboard": get_leaderboard,
}

GEMINI_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="query_assets",
            description="Query ad assets with exact filters and sorting. Use this for questions like 'show me TAB ads under $2 CPI' or 'what are the best werewolf hooks by writer X'. Returns structured results with ad codes, metrics, and openings.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ip": types.Schema(type="STRING", description="Show name or abbreviation: TAB, TOLR, WBM, AQB, M3VW, TAM, etc."),
                    "genre": types.Schema(type="STRING", description="Genre: Werewolf, Romance, Fantasy, etc."),
                    "writer": types.Schema(type="STRING", description="Writer name (partial match)"),
                    "style": types.Schema(type="STRING", description="Style filter (partial match)"),
                    "max_cpi": types.Schema(type="NUMBER", description="Maximum CPI threshold"),
                    "min_cpi": types.Schema(type="NUMBER", description="Minimum CPI threshold"),
                    "growth_only": types.Schema(type="BOOLEAN", description="Only return assets promoted to growth"),
                    "sort_by": types.Schema(type="STRING", description="Sort: 'cpi' (asc), 'cpi_desc', 'ctr_cti' (desc), 'total_spend' (desc)"),
                    "limit": types.Schema(type="INTEGER", description="Max results (default 20, max 50)"),
                    "search_text": types.Schema(type="STRING", description="Text search across opening, script_name, ad_name"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_asset_detail",
            description="Get full detail for a specific ad code including the complete opening text, full script text, and all performance metrics. Use when the user asks about a specific ad code.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ad_code": types.Schema(type="STRING", description="The ad code to look up (e.g., 'MAY25885')"),
                },
                required=["ad_code"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_show_context",
            description="Get show-level context: 10-hour base story, character canvas, CPI-cracking notes. Use when writing scripts for a specific show — this gives you the actual characters, world, and story to work with.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "show": types.Schema(type="STRING", description="Show name or abbreviation: TAB, TOLR, WBM, AQB, M3VW, etc."),
                    "section": types.Schema(type="STRING", description="Section: 'all', 'base_story', 'cpi_crack', 'character_canvas', 'hub'"),
                    "max_chars": types.Schema(type="INTEGER", description="Max chars to return (default 12000)"),
                },
                required=["show"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_opening_stats",
            description="Get stats for a specific opening_code (reuse count, avg CPI, all ad variants), or get the top most-reused opening codes. Use to find proven openings that have been successfully reused many times.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "opening_code": types.Schema(type="STRING", description="Specific opening code to look up. If omitted, returns top reused openings."),
                    "top_n_reused": types.Schema(type="INTEGER", description="How many top reused openings to return (default 20)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_writer_stats",
            description="Get performance stats for a specific writer (portfolio, avg CPI, top ads) or a leaderboard of top writers by avg CPI.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "writer": types.Schema(type="STRING", description="Writer name (partial match). If omitted, returns writer leaderboard."),
                    "top_n": types.Schema(type="INTEGER", description="How many writers for leaderboard (default 15)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_leaderboard",
            description="Get top N assets by any metric. Use for questions like 'top 10 by CPI', 'best CTR assets', 'highest spend ads'. Can filter by show or genre.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "metric": types.Schema(type="STRING", description="Metric: 'cpi' (lowest=best), 'ctr_cti' (highest=best), 'retention' (highest), 'spend' (highest), 'cpm' (lowest)"),
                    "ip": types.Schema(type="STRING", description="Show filter: TAB, TOLR, WBM, etc."),
                    "genre": types.Schema(type="STRING", description="Genre filter"),
                    "growth_only": types.Schema(type="BOOLEAN", description="Only growth-promoted assets"),
                    "limit": types.Schema(type="INTEGER", description="Max results (default 25)"),
                },
            ),
        ),
    ])
]


def execute_tool_call(func_name, args):
    """Execute a tool function and return JSON-serializable result."""
    fn = TOOL_FUNCTIONS.get(func_name)
    if not fn:
        return {"error": f"Unknown tool: {func_name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"Tool {func_name} failed: {str(e)}"}


def _truncate_result(obj, max_chars=24000):
    """Truncate a tool result dict so its JSON serialization stays under max_chars.
    Truncates string values and trims list items — always returns valid JSON-serializable data."""
    serialized = json.dumps(obj, default=str)
    if len(serialized) <= max_chars:
        return obj
    # Deep-truncate: shorten long string fields, trim list lengths
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and len(v) > 2000:
                out[k] = v[:2000] + "...[truncated]"
            elif isinstance(v, list) and len(v) > 15:
                out[k] = [_truncate_result(item, max_chars=1500) for item in v[:15]]
                out[k].append({"note": f"truncated — {len(v)} total items, showing first 15"})
            elif isinstance(v, dict):
                out[k] = _truncate_result(v, max_chars=3000)
            else:
                out[k] = v
        return out
    elif isinstance(obj, list):
        return [_truncate_result(item, max_chars=1500) for item in obj[:15]]
    elif isinstance(obj, str) and len(obj) > 2000:
        return obj[:2000] + "...[truncated]"
    return obj


def run_tool_calling_loop(gclient, prompt, system_instruction, stats_block, max_rounds=4):
    """
    Run Gemini with function calling in a loop:
    1. Send the user prompt + tools
    2. If model calls tools, execute them and send results back
    3. Repeat until model produces a final text response (or max rounds)

    Returns (final_text, tool_calls_log).
    """
    messages = [
        types.Content(role="user", parts=[types.Part(text=f"{stats_block}\n\nUSER REQUEST:\n{prompt}")]),
    ]
    tool_log = []

    for round_num in range(max_rounds):
        response = gclient.models.generate_content(
            model=CHAT_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0,
                tools=GEMINI_TOOLS,
            ),
        )

        # Check if model wants to call functions
        func_calls = []
        text_parts = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    func_calls.append(part.function_call)
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

        if not func_calls:
            # Model produced a final text response
            return "".join(text_parts), tool_log

        # Execute all function calls
        messages.append(response.candidates[0].content)
        func_response_parts = []
        for fc in func_calls:
            args = dict(fc.args) if fc.args else {}
            tool_log.append({"tool": fc.name, "args": args})
            result = execute_tool_call(fc.name, args)
            # Cap tool results to prevent context overflow — truncate data, not JSON
            result = _truncate_result(result, max_chars=24000)
            func_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response=result,
                )
            )

        messages.append(types.Content(role="user", parts=func_response_parts))

    # Exhausted rounds — return whatever text we got
    return "".join(text_parts) if text_parts else "(Tool calling exceeded max rounds)", tool_log


# ---------- two-pass generation ----------

CRITIQUE_PROMPT = """You are a ruthless creative director reviewing a draft Q1 opening script.

Evaluate the draft against these non-negotiable criteria. For each, score PASS or FAIL with a one-line reason:

1. BEAT 1 — SHOCK HOOK: Is the first sentence under 15 words? Is it a visceral, specific, filmable image (not a question, not exposition)?
2. BEAT 2 — TRAGIC BACKSTORY: Is the protagonist at her lowest status? Are the conditions specific and concrete?
3. BEAT 3 — ESCALATING NAMED ABUSE: Are antagonists named individually? Do they commit specific cruelties? Is there embedded quotable dialogue? Are there 3+ escalations?
4. BEAT 4 — IDENTITY REVEAL: Is there a supernatural power/identity moment? Is it mysterious, not over-explained?
5. BEAT 5 — FATED MATE + CLIFFHANGER: Does it end with an unresolved encounter? Will the viewer NEED to know what happens next?
6. WORD COUNT: Is it 430-550 words?
7. PROTAGONIST: Is the lead female?
8. VOICE: Is it first-person or intimate third-person with internal thoughts? Not distant literary prose?
9. PACING: Does every beat move in 3-4 sentences max? No scene lingers?
10. DIALOGUE: Is there embedded dialogue from antagonists that's cruel enough to screenshot?

For each FAIL, write a specific rewrite instruction (not vague — tell the writer exactly what to change).

End with: VERDICT: PASS (ready to test) or REWRITE NEEDED (with numbered fixes)."""


def two_pass_generate(gclient, prompt, system_instruction, stats_block, progress_callback=None):
    """
    Two-pass generation:
    1. Generate draft Q1 using tool calling
    2. Self-critique against the 5-beat formula
    3. If critique says REWRITE NEEDED, generate a revised version

    Returns (final_text, draft_text, critique_text, tool_log).
    """
    # Pass 1: Draft
    if progress_callback:
        progress_callback("Pass 1: Generating draft…")
    draft, tool_log = run_tool_calling_loop(gclient, prompt, system_instruction, stats_block)

    # Pass 2: Critique
    if progress_callback:
        progress_callback("Pass 2: Self-critiquing draft…")
    critique_input = f"DRAFT Q1 TO REVIEW:\n\n{draft}\n\nReview this draft against the criteria."
    critique_resp = gclient.models.generate_content(
        model=CHAT_MODEL,
        contents=critique_input,
        config=types.GenerateContentConfig(
            system_instruction=CRITIQUE_PROMPT,
            temperature=0.3,
        ),
    )
    critique = critique_resp.text or "(no critique generated)"

    # Check if rewrite needed
    if "REWRITE NEEDED" in critique.upper():
        if progress_callback:
            progress_callback("Pass 3: Rewriting based on critique…")
        rewrite_prompt = (
            f"You wrote this draft Q1:\n\n{draft}\n\n"
            f"Your creative director reviewed it and found these issues:\n\n{critique}\n\n"
            f"Now rewrite the Q1 fixing every issue flagged as FAIL. Keep everything that passed. "
            f"Return ONLY the revised Q1 script, nothing else."
        )
        rewrite_resp = gclient.models.generate_content(
            model=CHAT_MODEL,
            contents=rewrite_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0,
            ),
        )
        final = rewrite_resp.text or draft
        return final, draft, critique, tool_log
    else:
        return draft, draft, critique, tool_log


# ---------- file extraction ----------

def extract_file_text(uploaded_file):
    """Extract text from an uploaded file. Returns (text, file_type)."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith((".txt", ".md", ".csv", ".srt")):
        return data.decode("utf-8", errors="ignore"), "text"

    elif name.endswith((".docx",)):
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs), "document"
        except ImportError:
            return "(python-docx not installed)", "error"

    elif name.endswith((".pdf",)):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(p.extract_text() or "" for p in reader.pages), "document"
        except ImportError:
            return "(pypdf not installed)", "error"

    elif name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        import base64
        b64 = base64.b64encode(data).decode()
        return f"[Image: {uploaded_file.name}, {len(data)} bytes, base64 available]", "image"

    elif name.endswith((".json",)):
        return data.decode("utf-8", errors="ignore"), "json"

    elif name.endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
            import io
            df = pd.read_excel(io.BytesIO(data))
            return df.to_csv(index=False), "spreadsheet"
        except Exception as e:
            return f"(failed to read Excel: {e})", "error"

    else:
        return data.decode("utf-8", errors="ignore")[:10000], "unknown"


# ---------- feedback persistence ----------

FEEDBACK_PATH = Path(".tmp/feedback.json")


def load_feedback():
    if FEEDBACK_PATH.exists():
        try:
            return json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            pass
    return []


def save_feedback_entry(entry):
    fb = load_feedback()
    fb.append(entry)
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(fb, indent=2, default=str))


# ---------- project persistence ----------

PROJECTS_DIR = Path(".tmp/projects")


def load_projects():
    if not PROJECTS_DIR.exists():
        return {}
    out = {}
    for p in sorted(PROJECTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text())
            out[p.stem] = data
        except Exception:
            pass
    return out


def save_project(project_id, project):
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECTS_DIR / f"{project_id}.json").write_text(json.dumps(project, default=str))


def delete_project(project_id):
    p = PROJECTS_DIR / f"{project_id}.json"
    if p.exists():
        p.unlink()


def new_project_id():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(2).hex()


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
    /* ═══════════════════════════════════════════
       Stable, clean design — works WITH Streamlit, not against it.
       Uses config.toml for sidebar color + primary color.
       CSS only for what Streamlit can't theme natively.
       ═══════════════════════════════════════════ */

    /* Hide chrome */
    #MainMenu, footer { display: none !important; }

    /* Sidebar text colors on dark background (bg set via config.toml) */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #ececf1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #8e8ea0 !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: #ececf1 !important;
        text-align: left !important;
        border-radius: 0.5rem !important;
        font-size: 0.875rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #2a2b32 !important;
    }
    .new-chat-btn .stButton > button {
        border: 1px solid #565869 !important;
        font-weight: 500 !important;
    }

    /* Sidebar inputs on dark bg */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #40414f !important;
        color: #ececf1 !important;
        border-color: #565869 !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: #8e8ea0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
        color: #c5c5d2 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #353740 !important;
    }
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #acacbe !important;
        font-size: 0.8rem !important;
    }

    /* Chat input — visible text, clean border */
    [data-testid="stChatInput"] textarea {
        color: #353740 !important;
        background: #ffffff !important;
        border: 1px solid #d9d9e3 !important;
        border-radius: 1.25rem !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16,163,127,0.15) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8e8ea0 !important;
    }

    /* Chat message text */
    [data-testid="stChatMessageContent"] {
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        color: #353740 !important;
    }

    /* User messages — subtle background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f7f7f8 !important;
        border-radius: 0.75rem !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
        color: #8e8ea0 !important;
    }

    /* Empty state */
    .empty-hero { text-align: center; padding: 12vh 1rem 2rem 1rem; }
    .empty-hero h1 { font-size: 1.75rem; font-weight: 700; color: #353740; margin-bottom: 0.3rem; }
    .empty-hero p { color: #8e8ea0; font-size: 0.95rem; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #c5c5d2; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- init clients + state ----------

gclient = get_gemini()
show_catalog = get_show_catalog()
stats = get_dataset_stats()
dropdown_values = unique_ip_values()
stats_block = format_stats_block(stats)

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if "projects" not in st.session_state:
    st.session_state.projects = load_projects()
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None
if "settings" not in st.session_state:
    st.session_state.settings = {
        "two_pass": True,
    }
if "uploaded_context" not in st.session_state:
    st.session_state.uploaded_context = []


def get_current_project():
    pid = st.session_state.current_project_id
    if pid and pid in st.session_state.projects:
        return st.session_state.projects[pid]
    return None


def get_current_chat():
    cid = st.session_state.current_chat_id
    if cid and cid in st.session_state.chats:
        return st.session_state.chats[cid]
    return None


def start_new_chat(project_id=None):
    cid = new_chat_id()
    st.session_state.current_chat_id = cid
    st.session_state.chats[cid] = {
        "id": cid,
        "title": "New chat",
        "messages": [],
        "filters": {},
        "project_id": project_id,
    }
    st.session_state.uploaded_context = []


# ---------- sidebar ----------

with st.sidebar:
    st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#ececf1;margin:0.5rem 0 1rem 0;">🎬 Hook Lab</p>', unsafe_allow_html=True)

    # ── Projects ──
    with st.expander("📁  Projects", expanded=True):
        # Create new project
        cols_proj = st.columns([0.75, 0.25])
        with cols_proj[0]:
            new_proj_name = st.text_input("New project", placeholder="e.g. TAB Q1 Campaign", label_visibility="collapsed")
        with cols_proj[1]:
            if st.button("Create", key="create_proj", use_container_width=True) and new_proj_name.strip():
                pid = new_project_id()
                st.session_state.projects[pid] = {
                    "id": pid,
                    "name": new_proj_name.strip(),
                    "chat_ids": [],
                    "notes": "",
                }
                save_project(pid, st.session_state.projects[pid])
                st.session_state.current_project_id = pid
                st.rerun()

        # Project list
        cur_proj = get_current_project()
        proj_names = ["All chats"] + [p["name"] for p in st.session_state.projects.values()]
        proj_ids = [None] + list(st.session_state.projects.keys())
        cur_idx = proj_ids.index(st.session_state.current_project_id) if st.session_state.current_project_id in proj_ids else 0
        sel = st.selectbox("Active project", proj_names, index=cur_idx, label_visibility="collapsed")
        st.session_state.current_project_id = proj_ids[proj_names.index(sel)]

    # ── New chat button ──
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✏️  New chat", use_container_width=True, key="new_chat_btn"):
        start_new_chat(project_id=st.session_state.current_project_id)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat history (filtered by project) ──
    st.markdown('<p style="font-size:0.75rem;font-weight:500;color:#8e8ea0;text-transform:uppercase;letter-spacing:0.05em;margin:0.75rem 0 0.4rem 0;">Recent</p>', unsafe_allow_html=True)
    active_project = st.session_state.current_project_id
    filtered_chats = [
        (cid, chat) for cid, chat in st.session_state.chats.items()
        if active_project is None or chat.get("project_id") == active_project
    ]
    if not filtered_chats:
        st.caption("No chats yet.")
    else:
        for cid, chat in filtered_chats[:30]:
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

    # ── Settings ──
    with st.expander("⚙️  Settings", expanded=False):
        st.session_state.settings["two_pass"] = st.checkbox(
            "Two-pass generation (draft → critique → rewrite)",
            value=st.session_state.settings.get("two_pass", True),
        )

    # ── File upload ──
    with st.expander("📎  Attach files", expanded=False):
        uploaded = st.file_uploader(
            "Upload docs, images, scripts",
            accept_multiple_files=True,
            type=["txt", "md", "csv", "json", "docx", "pdf", "xlsx", "png", "jpg", "jpeg", "gif", "webp", "srt"],
            label_visibility="collapsed",
        )
        if uploaded:
            new_context = []
            for f in uploaded:
                text, ftype = extract_file_text(f)
                new_context.append({"name": f.name, "type": ftype, "text": text[:15000]})
            st.session_state.uploaded_context = new_context
            st.caption(f"{len(new_context)} file(s) attached")
            for fc in new_context:
                st.caption(f"  {fc['name']} ({fc['type']}, {len(fc['text']):,} chars)")
        if st.button("Clear attachments", key="clear_attach"):
            st.session_state.uploaded_context = []
            st.rerun()

    st.divider()
    brief_count = len(_load_briefs())
    st.caption(f"📚 {stats['total_tests']} ad tests loaded")
    if brief_count:
        st.caption(f"🎬 {brief_count} show briefs · {len(show_catalog)} shows")
    fb_count = len(load_feedback())
    if fb_count:
        st.caption(f"📊 {fb_count} feedback entries logged")


# ---------- main area ----------

current_chat = get_current_chat()

if current_chat is None or not current_chat.get("messages"):
    # Empty state
    st.markdown(
        '<div class="empty-hero">'
        '<h1>Hook Lab</h1>'
        '<p>Data-backed ad script generation for Pocket FM</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        ("🏆", "Top performers", "Show me the 10 best hooks and what makes them crack"),
        ("✍️", "Write a Q1", "Write a 2-min opening script for The Alpha's Bride under $2.50 CPI"),
        ("🔍", "Diagnose a draft", "Here's my draft hook: [paste]. What's wrong and how to fix it?"),
        ("📊", "Pattern analysis", "What do all sub-$2.00 CPI hooks share in the first 3 seconds?"),
        ("🎬", "Show deep-dive", "Break down My Three Vampire Wives — world, characters, best angles"),
        ("👤", "Writer stats", "Which writers hit under $2.50 CPI consistently? Show their best work"),
    ]

    cols = st.columns(2)
    for i, (icon, title, prompt_text) in enumerate(suggestions):
        with cols[i % 2]:
            st.markdown(
                f'<div style="border:1px solid #d9d9e3;border-radius:0.75rem;padding:0.85rem 1rem;margin-bottom:0.5rem;cursor:pointer;transition:background 0.15s;">'
                f'<div style="font-size:0.85rem;font-weight:600;color:#353740;margin-bottom:0.2rem;">{icon} {title}</div>'
                f'<div style="font-size:0.8rem;color:#8e8ea0;line-height:1.4;">{prompt_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Use this", key=f"sug_{i}", use_container_width=True):
                if current_chat is None:
                    start_new_chat(project_id=st.session_state.current_project_id)
                    current_chat = get_current_chat()
                st.session_state["pending_prompt"] = prompt_text
                st.rerun()
else:
    # Render conversation
    for msg_idx, msg in enumerate(current_chat["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # ── Tool log ──
                if msg.get("tool_log"):
                    with st.expander(f"🔧 Tools called ({len(msg['tool_log'])})", expanded=False):
                        for tc in msg["tool_log"]:
                            args_str = ", ".join(f"{k}={v!r}" for k, v in tc.get("args", {}).items())
                            st.markdown(f"- `{tc['tool']}({args_str})`")

                # ── Draft + critique (two-pass) ──
                if msg.get("draft") and msg.get("critique"):
                    with st.expander("📝 Draft + Critique (two-pass)", expanded=False):
                        st.markdown("**Draft:**")
                        st.markdown(msg["draft"][:2000])
                        st.markdown("---")
                        st.markdown("**Critique:**")
                        st.markdown(msg["critique"][:2000])

                # ── Feedback buttons ──
                fb_key = f"fb_{current_chat['id']}_{msg_idx}"
                existing_fb = msg.get("feedback")

                if existing_fb:
                    icon = "👍" if existing_fb.get("rating") == "up" else "👎"
                    cpi_val = existing_fb.get("actual_cpi")
                    st.caption(f"{icon} Feedback logged{f' — actual CPI: ${cpi_val}' if cpi_val else ''}")
                else:
                    fb_cols = st.columns([0.08, 0.08, 0.3, 0.15, 0.39])
                    with fb_cols[0]:
                        if st.button("👍", key=f"{fb_key}_up"):
                            msg["feedback"] = {"rating": "up", "actual_cpi": None, "notes": ""}
                            save_feedback_entry({
                                "chat_id": current_chat["id"],
                                "msg_idx": msg_idx,
                                "rating": "up",
                                "prompt": current_chat["messages"][msg_idx - 1]["content"] if msg_idx > 0 else "",
                                "response_preview": msg["content"][:500],
                            })
                            save_chat(current_chat["id"], current_chat)
                            st.rerun()
                    with fb_cols[1]:
                        if st.button("👎", key=f"{fb_key}_down"):
                            msg["feedback"] = {"rating": "down", "actual_cpi": None, "notes": ""}
                            save_feedback_entry({
                                "chat_id": current_chat["id"],
                                "msg_idx": msg_idx,
                                "rating": "down",
                                "prompt": current_chat["messages"][msg_idx - 1]["content"] if msg_idx > 0 else "",
                                "response_preview": msg["content"][:500],
                            })
                            save_chat(current_chat["id"], current_chat)
                            st.rerun()
                    with fb_cols[2]:
                        cpi_input = st.number_input(
                            "Actual CPI", min_value=0.0, value=0.0, step=0.1,
                            key=f"{fb_key}_cpi", label_visibility="collapsed",
                        )
                    with fb_cols[3]:
                        if st.button("Log CPI", key=f"{fb_key}_logcpi") and cpi_input > 0:
                            msg["feedback"] = {"rating": "logged", "actual_cpi": cpi_input, "notes": ""}
                            save_feedback_entry({
                                "chat_id": current_chat["id"],
                                "msg_idx": msg_idx,
                                "rating": "cpi_logged",
                                "actual_cpi": cpi_input,
                                "prompt": current_chat["messages"][msg_idx - 1]["content"] if msg_idx > 0 else "",
                                "response_preview": msg["content"][:500],
                            })
                            save_chat(current_chat["id"], current_chat)
                            st.rerun()

                # Legacy support
                if msg.get("sources"):
                    with st.expander(f"📚 Ad sources ({len(msg['sources'])})", expanded=False):
                        for h in msg["sources"]:
                            m = h.get("meta", {})
                            cpi = f"${m['cpi']:.2f}" if m.get("cpi", -1) >= 0 else "?"
                            st.markdown(f"- **{m.get('ad_code','?')}** — CPI {cpi}")


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
        placeholder = st.empty()
        full_text = ""
        draft_text = ""
        critique_text = ""
        tool_log = []

        # Build prompt with file context if any
        full_prompt = prompt
        if st.session_state.uploaded_context:
            file_ctx = "\n\n".join(
                f"[ATTACHED FILE: {fc['name']} ({fc['type']})]\n{fc['text']}"
                for fc in st.session_state.uploaded_context
            )
            full_prompt = f"{file_ctx}\n\n{prompt}"

        try:
            use_two_pass = st.session_state.settings.get("two_pass", True)

            # Detect if this is a Q1/opening generation request (two-pass worthy)
            generation_keywords = ["write", "generate", "create", "give me", "q1", "opening", "hook", "script", "draft"]
            is_generation = any(kw in prompt.lower() for kw in generation_keywords)

            if use_two_pass and is_generation:
                status_placeholder = st.empty()
                def update_status(msg):
                    status_placeholder.caption(f"⏳ {msg}")

                full_text, draft_text, critique_text, tool_log = two_pass_generate(
                    gclient, full_prompt, SYSTEM_PROMPT, stats_block,
                    progress_callback=update_status,
                )
                status_placeholder.empty()
            else:
                with st.spinner("Thinking + querying data…"):
                    full_text, tool_log = run_tool_calling_loop(
                        gclient, full_prompt, SYSTEM_PROMPT, stats_block, max_rounds=4,
                    )

            placeholder.markdown(full_text)
        except Exception as e:
            full_text = f"⚠️ Generation failed: {e}"
            placeholder.error(full_text)

        # Show tool calls
        if tool_log:
            with st.expander(f"🔧 Tools called ({len(tool_log)})", expanded=False):
                for tc in tool_log:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    st.markdown(f"- `{tc['tool']}({args_str})`")

        # Show draft + critique if two-pass was used
        if draft_text and critique_text and draft_text != full_text:
            with st.expander("📝 Draft + Critique (two-pass)", expanded=False):
                st.markdown("**Draft:**")
                st.markdown(draft_text[:2000])
                st.markdown("---")
                st.markdown("**Critique:**")
                st.markdown(critique_text[:2000])

        # Show attached files
        if st.session_state.uploaded_context:
            with st.expander(f"📎 Files used ({len(st.session_state.uploaded_context)})", expanded=False):
                for fc in st.session_state.uploaded_context:
                    st.caption(f"{fc['name']} ({fc['type']}, {len(fc['text']):,} chars)")

    msg_data = {
        "role": "assistant",
        "content": full_text,
        "tool_log": tool_log,
    }
    if draft_text and draft_text != full_text:
        msg_data["draft"] = draft_text
        msg_data["critique"] = critique_text
    if st.session_state.uploaded_context:
        msg_data["files_used"] = [fc["name"] for fc in st.session_state.uploaded_context]

    current_chat["messages"].append(msg_data)
    if current_chat.get("title") == "New chat":
        current_chat["title"] = title_from_first_message(current_chat["messages"])
    save_chat(current_chat["id"], current_chat)
    st.rerun()
