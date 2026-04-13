"""
FastAPI backend for Superbot.
Wraps existing data_tools + Gemini function calling.
Serves the static frontend + handles chat API.
"""
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import io

load_dotenv()

# Add tools/ to path so we can import data_tools
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from data_tools import (
    query_assets, get_asset_detail, get_show_context,
    get_opening_stats, get_writer_stats, get_leaderboard,
    _load_assets, _load_briefs, SHOW_ALIASES,
)

from google import genai
from google.genai import types

# ── Config ──
CHAT_MODEL = "gemini-2.5-flash"
CHATS_DIR = Path(".tmp/chats")
PROJECTS_DIR = Path(".tmp/projects")
FEEDBACK_PATH = Path(".tmp/feedback.json")

# ── Load system prompt from chatbot_app.py ──
# We import the SYSTEM_PROMPT constant
sys.path.insert(0, str(Path(__file__).parent / "tools"))
_chatbot_mod = {}
exec(
    compile(
        "SYSTEM_PROMPT = None\n" +
        open(Path(__file__).parent / "tools" / "chatbot_app.py").read().split("SYSTEM_PROMPT = ")[1].split('\n\n\n# ----------')[0],
        "<system_prompt>", "exec"
    ),
    _chatbot_mod
)

# Actually, let's just read it directly with a simpler approach
def _extract_system_prompt():
    src = (Path(__file__).parent / "tools" / "chatbot_app.py").read_text()
    start = src.find('SYSTEM_PROMPT = """') + len('SYSTEM_PROMPT = """')
    end = src.find('"""', start)
    return src[start:end]

_BASE_SYSTEM_PROMPT = _extract_system_prompt()

# Add table formatting + CPI formula knowledge
SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + """

====================================================================
RESPONSE FORMATTING
====================================================================

When the user asks for data, analytics, comparisons, or lists — ALWAYS use MARKDOWN TABLES.
- Use | Column | Column | format with header separators
- Sort by the most relevant metric (usually CPI ascending)
- Include ad_code, show, writer, CPI, CTR*CTI, and any other relevant columns
- Keep tables clean and scannable

When the user asks for analysis or recommendations (not scripts), respond with structured data — NOT a script critique.

====================================================================
CPI FORMULA — CORE METRIC
====================================================================

CPI = CPM / (CTR × CTI)

- CPM = Cost Per Mille (cost per 1,000 impressions) — we want this LOW
- CTR = Click-Through Rate — we want this HIGH (the ad must make people CLICK)
- CTI = Click-To-Install rate — we want this HIGH (the landing page must convert)
- CTR × CTI = the conversion chain. This is what the CREATIVE controls.
- A great opening/hook drives CTR. A great cliffhanger drives CTI.
- Synergy between opening and cliffhanger = maximum CTR × CTI = lowest CPI.
"""

# ── Gemini client ──
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not set")
gclient = genai.Client(api_key=api_key)

# ── Dataset stats (computed once) ──
from collections import defaultdict

def compute_stats():
    assets = _load_assets()
    with_cpi = [a for a in assets if a.get("cpi") and a["cpi"] > 0]
    top_cpi = sorted(with_cpi, key=lambda a: a["cpi"])[:25]
    with_ctr = [a for a in assets if a.get("ctr_cti") and a["ctr_cti"] > 0]
    top_ctr = sorted(with_ctr, key=lambda a: -a["ctr_cti"])[:25]
    scaled = sorted([a for a in with_cpi if a.get("is_active_growth")], key=lambda a: a["cpi"])[:25]
    by_ip = defaultdict(list)
    for a in with_cpi:
        if a.get("ip"):
            by_ip[a["ip"]].append(a["cpi"])
    ip_leader = sorted(
        [(ip, sum(v)/len(v), len(v)) for ip, v in by_ip.items() if len(v) >= 3],
        key=lambda x: x[1]
    )[:15]
    by_writer = defaultdict(list)
    for a in with_cpi:
        if a.get("writer"):
            by_writer[a["writer"]].append(a["cpi"])
    writer_leader = sorted(
        [(w, sum(v)/len(v), len(v)) for w, v in by_writer.items() if len(v) >= 5],
        key=lambda x: x[1]
    )[:15]

    lines = ["DATASET STATS:"]
    lines.append(f"Total: {len(assets)} assets, {len(with_cpi)} with CPI")
    lines.append("\nTOP 15 BY CPI:")
    for a in top_cpi[:15]:
        lines.append(f"  ${a['cpi']:.2f} {a.get('ad_code','?')} | {a.get('ip','?')} | \"{(a.get('opening') or '')[:120]}\"")
    lines.append("\nTOP WRITERS (avg CPI):")
    for w, avg, n in writer_leader[:10]:
        lines.append(f"  {w}: ${avg:.2f} ({n} ads)")
    lines.append("\nIP VOLUME:")
    for ip, cpis in sorted(by_ip.items(), key=lambda x: -len(x[1]))[:10]:
        lines.append(f"  {ip}: {len(cpis)} ads, avg ${sum(cpis)/len(cpis):.2f}")
    return "\n".join(lines)

STATS_BLOCK = compute_stats()

# ── Mode-specific instructions ──

MODE_INSTRUCTIONS = {
    "opening": """
====================================================================
MODE: OPENING (180-second hook content)
====================================================================

You are writing an OPENING — a ~180-second (roughly 650-800 words) piece of content designed to instantly hook viewers within the first 3 seconds and keep them watching.

KEY PRINCIPLES:
- The opening may or may NOT be related to the actual story. We often show something bizarre/extreme to grab attention, then merge seamlessly into the base story.
- Think of it as a "cold open" — something so outrageous the viewer cannot scroll past.
- CPI = CPM / (CTR × CTI). We need CPM low and CTR × CTI high. The opening's job is to MAXIMIZE CTR (click-through rate).
- The opening must be visceral, specific, filmable — not abstract or literary.

STRUCTURE:
1. HOOK (0-3 seconds): The most shocking, bizarre, or emotionally extreme image/line. Under 15 words.
2. ESCALATION (3-30 seconds): Build on the hook with escalating stakes. Specific, named characters. Quotable cruel dialogue.
3. TENSION PEAK (30-90 seconds): Maximum emotional intensity. The viewer is hooked — they MUST know what happens.
4. MERGE POINT (90-120 seconds): This is where the opening transitions into the base story. The merge must feel SEAMLESS — not jarring.
5. STORY ENTRY (120-180 seconds): We're now in the actual show's world. Characters, conflict, and stakes are established.

IMPORTANT — MERGE POINT:
- Always clearly mark where the merge happens: **[MERGE POINT — transition to base story]**
- Explain HOW this bizarre opening connects to the real story
- The emotional thread must carry through — if the opening is about betrayal, the story entry should also involve betrayal
- The viewer should NOT feel tricked — the merge should feel like "oh, THIS is why they showed me that"

CLIFFHANGER ALIGNMENT:
- The opening's emotional promise must align with the cliffhanger at the end of the full script
- If you open with betrayal, the cliffhanger should pay off or escalate that betrayal
- Synergy between opening hook and ending cliffhanger = higher CTI (click-to-install)

OUTPUT: Provide the full opening script with the merge point clearly marked. After the script, add a brief note explaining:
- Why this opening works (what proven pattern it mirrors)
- Where exactly the merge point is and why it works there
- How it aligns with the recommended cliffhanger
""",

    "q1": """
====================================================================
MODE: Q1 SCRIPT (First Quarter — ~2 minutes, 450-530 words)
====================================================================

You are writing a Q1 SCRIPT — the first quarter of an ad script (~450-530 words, ~2 minutes of narration).

CPI = CPM / (CTR × CTI). The Q1's job is to build CONVICTION and CONNECTION with the female lead so strong that the viewer installs the app.

KEY PRINCIPLES:
- This introduces the main character and makes the viewer CARE about her
- Every line must build emotional investment — if a line doesn't make the viewer feel something, cut it
- The Q1 must be hooky AND fundamentally aligned with the show's actual story
- Female protagonist MANDATORY — every sub-$2.50 winner centers a woman 18-35 audience can identify with

THE 5-BEAT FORMULA (from analyzing every sub-$2.00 CPI asset):
BEAT 1: SHOCK HOOK (1-2 sentences, under 15 words) — visceral, specific, filmable
BEAT 2: TRAGIC BACKSTORY (3-5 sentences) — protagonist at absolute lowest status
BEAT 3: ESCALATING NAMED ABUSE (5-8 sentences, longest beat) — named antagonists, specific cruelties, quotable dialogue, 3-4 escalations
BEAT 4: SUPERNATURAL IDENTITY REVEAL (2-4 sentences) — power awakening, mysterious
BEAT 5: FATED MATE ENCOUNTER + CLIFFHANGER (3-5 sentences) — unresolved tension, viewer MUST know what happens next

CONVICTION + CONNECTION:
- The viewer must feel the protagonist's pain as their OWN
- Use first-person voice or deeply intimate third-person
- Name every character — never "they" or "the man"
- Every antagonist line should be cruel enough to screenshot
- The cliffhanger must create an unbearable need to know what happens next
""",

    "base": """
====================================================================
MODE: BASE SCRIPT (Full standalone 8-12 minute script, ~5,500-6,000 words)
====================================================================

You are writing a FULL BASE SCRIPT — a standalone 8-12 minute ad script (~5,500-6,000 words).

This is the complete script from Q1 through Q4 + CTA. Structure:
- Q1 (2 min): Hook + character intro + conviction building (see Q1 formula)
- Q2 (3 min): Deepen conflict, introduce love interest, raise stakes
- Q3 (3 min): Climax — maximum danger/tension, power reveal, confrontation
- Q4 (2 min): Resolution tease + massive cliffhanger + CTA

CPI = CPM / (CTR × CTI). The full script's job is to maximize CTI — the viewer must be SO invested they install the app.

CLIFFHANGER (final 30 seconds):
- Must be the most emotionally intense moment in the entire script
- Leave the viewer with an UNRESOLVED question they cannot ignore
- Align with the hook — the emotional promise of the opening must pay off here
- End with CTA: "To find out what happens next, listen to [SHOW NAME] — only on Pocket FM. Download free."
""",

    "merge": """
====================================================================
MODE: SCRIPT MERGE (Combining two scripts into one)
====================================================================

You are MERGING two scripts together. Before writing, you MUST ask clarifying questions if any of these are unclear:
- Which scripts are being merged? (provide ad codes or paste the scripts)
- What is the target length? (shorter combined version or longer combined version?)
- Which opening should be used? (from script A, script B, or a new one?)
- What is the primary show/IP?
- Should the merge prioritize one script's story arc over the other?

MERGE PRINCIPLES:
- Identify the strongest elements from each script (best hook, best escalation, best cliffhanger)
- The merge must feel like ONE coherent story, not two stories stitched together
- Transition points between source material must be seamless
- Maintain consistent voice, tense, and character names throughout
- The merged cliffhanger should be the strongest ending from either source, or an even stronger combination

OUTPUT: The merged script + a brief note explaining what you took from each source and why.
""",

    "cliffhanger": """
====================================================================
MODE: CLIFFHANGER RE-WRITE (Improve CTR × CTI)
====================================================================

You are rewriting the CLIFFHANGER (final 30-60 seconds) of an existing script to improve CTR × CTI.

CPI = CPM / (CTR × CTI). The cliffhanger is the last thing the viewer sees before the CTA. It must create maximum unresolved tension — the viewer installs because they CANNOT not know what happens next.

WHAT MAKES A KILLER CLIFFHANGER:
1. UNRESOLVED THREAT: Someone is in immediate danger — the scene cuts before we know if they survive
2. IDENTITY BOMB: A secret is about to be revealed — the scene cuts before we hear it
3. IMPOSSIBLE CHOICE: The protagonist must choose between two devastating options — we never see the choice
4. BETRAYAL MOMENT: Someone trusted turns — the scene cuts on the look of realization
5. POWER SURGE: The protagonist's hidden power activates — we see the beginning but not the outcome

REWRITE RULES:
- Read the existing cliffhanger and identify why it's weak (too resolved? too vague? no stakes?)
- The new cliffhanger must be SPECIFIC — not "something bad happened" but "Alpha Damon's claws were at her throat when her eyes turned silver"
- It must align with the opening hook — emotional synergy between start and end
- Keep it to 3-5 sentences max — punchy, not drawn out
- End with the CTA immediately after peak tension

OUTPUT: The rewritten cliffhanger + explanation of what was weak and what you fixed.
""",

    "super": """
====================================================================
MODE: SUPERAGENT (Full creative suite)
====================================================================

You are operating in SUPERAGENT mode — you can do EVERYTHING:
- Analyze data, find patterns, compare writers/shows/openings
- Write openings, Q1 scripts, full base scripts
- Merge scripts together
- Rewrite cliffhangers
- Provide strategic recommendations

CPI = CPM / (CTR × CTI). Every recommendation should be grounded in this formula.

IMPORTANT CONTEXT:
- Openings: ~180 seconds, may be bizarre/unrelated to story, must have clear merge point
- Q1: First quarter, ~450-530 words, builds conviction + connection with female lead
- Base Script: Full 8-12 min standalone, ~5,500-6,000 words
- Cliffhanger: Final 30-60 seconds, must maximize unresolved tension for CTI
- Script Merge: Combining scripts, ask clarifying questions if unclear

When the user's request spans multiple modes, handle them all. If something is unclear, ask before proceeding.
""",
}

# Mode-specific critique prompts
MODE_CRITIQUE = {
    "q1": """You are a ruthless creative director reviewing a draft Q1 opening script.
Evaluate against these non-negotiable criteria. For each, score PASS or FAIL with a one-line reason:

1. SHOCK HOOK: First sentence under 15 words? Visceral, specific, filmable image?
2. TRAGIC BACKSTORY: Protagonist at lowest status? Conditions specific and concrete?
3. NAMED ABUSE: Antagonists named individually? Specific cruelties? Quotable dialogue? 3+ escalations?
4. IDENTITY REVEAL: Supernatural power moment? Mysterious, not over-explained?
5. FATED MATE + CLIFFHANGER: Unresolved encounter? Viewer NEEDS to know what happens?
6. WORD COUNT: 430-550 words?
7. FEMALE PROTAGONIST?
8. VOICE: First-person/intimate? Not literary prose?
9. PACING: Every beat 3-4 sentences max?
10. DIALOGUE: Antagonist lines cruel enough to screenshot?

For each FAIL, write a specific rewrite instruction.
End with: VERDICT: PASS or REWRITE NEEDED with numbered fixes.""",

    "opening": """You are a ruthless creative director reviewing a draft OPENING (180-second hook content).
Evaluate against these criteria. For each, score PASS or FAIL with a one-line reason:

1. HOOK POWER: Does the first line stop the scroll? Under 15 words? Visceral/bizarre/extreme?
2. ESCALATION: Do the first 30 seconds build on the hook with rising stakes?
3. TENSION PEAK: Is there a clear maximum-intensity moment before the merge?
4. MERGE POINT: Is there a clearly marked merge point? Is the transition seamless?
5. STORY ENTRY: After the merge, are we grounded in real characters and story?
6. WORD COUNT: 650-800 words (~180 seconds)?
7. CTR POTENTIAL: Would YOU stop scrolling for this? Is it bizarre/extreme enough?
8. CLIFFHANGER ALIGNMENT: Does the opening's emotional promise connect to the ending?
9. VOICE: Visceral first-person or intimate? Not literary?
10. SPECIFICITY: Named characters? Filmable images? No vague abstractions?

For each FAIL, write a specific rewrite instruction.
End with: VERDICT: PASS or REWRITE NEEDED with numbered fixes.""",

    "base": """You are a ruthless creative director reviewing a full BASE SCRIPT (8-12 minute ad).
Evaluate against these criteria. For each, score PASS or FAIL:

1. Q1 HOOK: Does it open with a visceral, scroll-stopping hook?
2. CHARACTER ARC: Does the protagonist go from lowest to powerful across the script?
3. ESCALATION: Do stakes rise consistently through Q2-Q3?
4. LOVE INTEREST: Is the fated mate/love interest introduced with tension?
5. CLIMAX: Is Q3's confrontation emotionally intense and specific?
6. CLIFFHANGER: Does the ending create UNBEARABLE need to know what happens?
7. WORD COUNT: 5,000-6,500 words?
8. VOICE CONSISTENCY: First-person/intimate throughout? No drift to literary?
9. PACING: No scene lingers more than 5-6 sentences? Relentless momentum?
10. CTA: Does it end with clear install CTA after peak tension?

For each FAIL, write a specific rewrite instruction.
End with: VERDICT: PASS or REWRITE NEEDED with numbered fixes.""",

    "cliffhanger": """You are a ruthless creative director reviewing a CLIFFHANGER rewrite.
Evaluate against these criteria. For each, score PASS or FAIL:

1. UNRESOLVED TENSION: Does it leave something critically unresolved?
2. SPECIFICITY: Named characters? Specific actions? Not vague?
3. STAKES: Are the consequences clear and devastating?
4. EMOTIONAL PEAK: Is this the most intense moment in the script?
5. BREVITY: 3-5 sentences max? Punchy, not drawn out?
6. CTI POTENTIAL: Would the viewer install JUST to resolve this moment?
7. SYNERGY: Does it connect emotionally to the opening hook?

For each FAIL, write a specific rewrite instruction.
End with: VERDICT: PASS or REWRITE NEEDED with numbered fixes.""",
}

# ── Tool calling ──

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
            description="Query ad assets with exact filters. Use for 'show me TAB ads under $2 CPI'.",
            parameters=types.Schema(type="OBJECT", properties={
                "ip": types.Schema(type="STRING", description="Show: TAB, TOLR, WBM, AQB, M3VW, etc."),
                "genre": types.Schema(type="STRING"), "writer": types.Schema(type="STRING"),
                "max_cpi": types.Schema(type="NUMBER"), "min_cpi": types.Schema(type="NUMBER"),
                "growth_only": types.Schema(type="BOOLEAN"),
                "sort_by": types.Schema(type="STRING", description="cpi/cpi_desc/ctr_cti/total_spend"),
                "limit": types.Schema(type="INTEGER"), "search_text": types.Schema(type="STRING"),
            }),
        ),
        types.FunctionDeclaration(
            name="get_asset_detail",
            description="Full detail for a specific ad code — metrics + full script text.",
            parameters=types.Schema(type="OBJECT", properties={
                "ad_code": types.Schema(type="STRING"),
            }, required=["ad_code"]),
        ),
        types.FunctionDeclaration(
            name="get_show_context",
            description="Get show's 10HR base story, character canvas, CPI-cracking notes.",
            parameters=types.Schema(type="OBJECT", properties={
                "show": types.Schema(type="STRING"),
                "section": types.Schema(type="STRING", description="all/base_story/cpi_crack/character_canvas"),
                "max_chars": types.Schema(type="INTEGER"),
            }, required=["show"]),
        ),
        types.FunctionDeclaration(
            name="get_opening_stats",
            description="Stats for an opening_code (reuse count, avg CPI) or top most-reused openings.",
            parameters=types.Schema(type="OBJECT", properties={
                "opening_code": types.Schema(type="STRING"),
                "top_n_reused": types.Schema(type="INTEGER"),
            }),
        ),
        types.FunctionDeclaration(
            name="get_writer_stats",
            description="Writer portfolio or leaderboard by avg CPI.",
            parameters=types.Schema(type="OBJECT", properties={
                "writer": types.Schema(type="STRING"),
                "top_n": types.Schema(type="INTEGER"),
            }),
        ),
        types.FunctionDeclaration(
            name="get_leaderboard",
            description="Top N assets by metric (cpi/ctr_cti/retention/spend/cpm).",
            parameters=types.Schema(type="OBJECT", properties={
                "metric": types.Schema(type="STRING"),
                "ip": types.Schema(type="STRING"), "genre": types.Schema(type="STRING"),
                "growth_only": types.Schema(type="BOOLEAN"),
                "limit": types.Schema(type="INTEGER"),
            }),
        ),
    ])
]


def truncate_result(obj, max_chars=20000):
    serialized = json.dumps(obj, default=str)
    if len(serialized) <= max_chars:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and len(v) > 2000:
                out[k] = v[:2000] + "...[truncated]"
            elif isinstance(v, list) and len(v) > 12:
                out[k] = v[:12]
                out[k].append({"note": f"showing 12 of {len(v)}"})
            elif isinstance(v, dict):
                out[k] = truncate_result(v, 3000)
            else:
                out[k] = v
        return out
    return obj


def get_system_prompt_for_mode(mode):
    """Build system prompt with mode-specific instructions appended."""
    base = SYSTEM_PROMPT
    if mode and mode in MODE_INSTRUCTIONS:
        base += "\n\n" + MODE_INSTRUCTIONS[mode]
    return base


def get_critique_prompt_for_mode(mode):
    """Return the appropriate critique prompt for the mode, or None if no critique applies."""
    if mode in MODE_CRITIQUE:
        return MODE_CRITIQUE[mode]
    # For 'super' mode, use Q1 critique only if the draft looks like a script
    if mode == "super":
        return MODE_CRITIQUE["q1"]
    # For 'merge' mode, use base script critique
    if mode == "merge":
        return MODE_CRITIQUE["base"]
    return None


def should_critique(prompt, mode, draft):
    """Determine if this response should be critiqued."""
    if len(draft) < 200:
        return False
    # If mode is explicitly set to a script type, always critique
    if mode in ("q1", "opening", "base", "cliffhanger", "merge"):
        return True
    # For super mode or no mode, check if the response is actually a script
    if mode == "super" or mode is None:
        gen_keywords = ["write", "generate", "create", "give me", "q1", "opening", "hook", "script", "draft", "rewrite"]
        is_generation = any(kw in prompt.lower() for kw in gen_keywords)
        if not is_generation:
            return False
        # Check if draft looks like a data report (tables, lists of ad codes) vs a script
        data_indicators = ["| ad_code", "| cpi", "| writer", "here are the writers", "here are the top",
                           "here is a summary", "here is the data", "the following table"]
        is_data = any(ind in draft.lower() for ind in data_indicators)
        if is_data:
            return False
        return True
    return False


def run_generation(prompt, two_pass=True, mode=None):
    """Run tool-calling loop + optional mode-aware two-pass critique."""
    sys_prompt = get_system_prompt_for_mode(mode)
    messages = [
        types.Content(role="user", parts=[types.Part(text=f"{STATS_BLOCK}\n\nUSER REQUEST:\n{prompt}")]),
    ]
    tool_log = []

    # Tool calling loop
    text_parts = []
    for _ in range(4):
        response = gclient.models.generate_content(
            model=CHAT_MODEL, contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt, temperature=1.0, tools=GEMINI_TOOLS,
            ),
        )
        func_calls = []
        text_parts = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    func_calls.append(part.function_call)
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

        if not func_calls:
            break

        messages.append(response.candidates[0].content)
        func_response_parts = []
        for fc in func_calls:
            args = dict(fc.args) if fc.args else {}
            tool_log.append({"tool": fc.name, "args": args})
            fn = TOOL_FUNCTIONS.get(fc.name)
            result = fn(**args) if fn else {"error": f"Unknown: {fc.name}"}
            result = truncate_result(result)
            func_response_parts.append(
                types.Part.from_function_response(name=fc.name, response=result)
            )
        messages.append(types.Content(role="user", parts=func_response_parts))

    draft = "".join(text_parts)
    critique = None
    final = draft

    # Mode-aware two-pass critique
    if two_pass and should_critique(prompt, mode, draft):
        critique_sys = get_critique_prompt_for_mode(mode or "q1")
        if critique_sys:
            critique_input = f"DRAFT TO REVIEW:\n\n{draft}\n\nReview this draft against the criteria."
            cr = gclient.models.generate_content(
                model=CHAT_MODEL, contents=critique_input,
                config=types.GenerateContentConfig(system_instruction=critique_sys, temperature=0.3),
            )
            critique = cr.text or ""

            if "REWRITE NEEDED" in critique.upper():
                rw = gclient.models.generate_content(
                    model=CHAT_MODEL,
                    contents=f"Draft:\n{draft}\n\nCritique:\n{critique}\n\nRewrite fixing every FAIL. Return ONLY the revised script.",
                    config=types.GenerateContentConfig(system_instruction=sys_prompt, temperature=1.0),
                )
                final = rw.text or draft

    return {
        "response": final,
        "draft": draft if critique else None,
        "critique": critique,
        "tool_log": tool_log,
    }


# ── FastAPI app ──

app = FastAPI(title="Superbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static frontend
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    project_id: Optional[str] = None
    two_pass: bool = True
    file_context: Optional[str] = None
    mode: Optional[str] = None  # opening, q1, base, merge, cliffhanger, super, or None (auto)


class FeedbackRequest(BaseModel):
    chat_id: str
    msg_idx: int
    rating: str  # "up", "down", "cpi_logged"
    actual_cpi: Optional[float] = None


class ProjectRequest(BaseModel):
    name: str


# ── Login ──
VALID_USERS = {
    "werewolf": os.getenv("SUPERBOT_PASS", "pfmsuperbot@"),
}


@app.post("/api/login")
async def login(req: LoginRequest):
    expected = VALID_USERS.get(req.username.lower())
    if expected and req.password == expected:
        return {"ok": True}
    return JSONResponse({"ok": False, "error": "Invalid username or password"}, status_code=401)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    prompt = req.message
    if req.file_context:
        prompt = f"{req.file_context}\n\n{prompt}"
    try:
        result = run_generation(prompt, two_pass=req.two_pass, mode=req.mode)
    except Exception as e:
        raise HTTPException(500, str(e))

    # Save to chat history
    chat_id = req.chat_id or str(uuid.uuid4())
    chat_path = CHATS_DIR / f"{chat_id}.json"
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    if chat_path.exists():
        chat = json.loads(chat_path.read_text())
    else:
        chat = {"id": chat_id, "title": "", "messages": [], "project_id": req.project_id}
    chat["messages"].append({"role": "user", "content": req.message})
    chat["messages"].append({
        "role": "assistant", "content": result["response"],
        "tool_log": result["tool_log"],
        "draft": result.get("draft"), "critique": result.get("critique"),
    })
    if not chat["title"]:
        chat["title"] = req.message[:50]
    chat_path.write_text(json.dumps(chat, default=str))

    return {
        "chat_id": chat_id,
        "response": result["response"],
        "draft": result.get("draft"),
        "critique": result.get("critique"),
        "tool_log": result["tool_log"],
    }


@app.get("/api/chats")
async def list_chats(project_id: Optional[str] = None):
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chats = []
    for p in sorted(CHATS_DIR.glob("*.json"), reverse=True):
        try:
            c = json.loads(p.read_text())
            if project_id and c.get("project_id") != project_id:
                continue
            chats.append({"id": c["id"], "title": c.get("title", ""), "project_id": c.get("project_id"), "msg_count": len(c.get("messages", []))})
        except Exception:
            pass
    return chats[:50]


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    p = CHATS_DIR / f"{chat_id}.json"
    if not p.exists():
        raise HTTPException(404, "Chat not found")
    return json.loads(p.read_text())


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    p = CHATS_DIR / f"{chat_id}.json"
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.get("/api/projects")
async def list_projects():
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    projects = []
    for p in sorted(PROJECTS_DIR.glob("*.json"), reverse=True):
        try:
            projects.append(json.loads(p.read_text()))
        except Exception:
            pass
    return projects


@app.post("/api/projects")
async def create_project(req: ProjectRequest):
    pid = datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(2).hex()
    proj = {"id": pid, "name": req.name}
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECTS_DIR / f"{pid}.json").write_text(json.dumps(proj))
    return proj


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    fb = []
    if FEEDBACK_PATH.exists():
        try:
            fb = json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            pass
    fb.append({
        "chat_id": req.chat_id, "msg_idx": req.msg_idx,
        "rating": req.rating, "actual_cpi": req.actual_cpi,
        "timestamp": datetime.now().isoformat(),
    })
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(fb, indent=2))
    return {"ok": True}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Extract text from an uploaded file."""
    data = await file.read()
    name = (file.filename or "unknown").lower()
    text = ""
    ftype = "unknown"

    if name.endswith((".txt", ".md", ".csv", ".srt")):
        text = data.decode("utf-8", errors="ignore")
        ftype = "text"
    elif name.endswith(".json"):
        text = data.decode("utf-8", errors="ignore")
        ftype = "json"
    elif name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
            ftype = "document"
        except Exception as e:
            text = f"(failed to parse docx: {e})"
            ftype = "error"
    elif name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            ftype = "document"
        except Exception as e:
            text = f"(failed to parse pdf: {e})"
            ftype = "error"
    elif name.endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(data))
            text = df.to_csv(index=False)
            ftype = "spreadsheet"
        except Exception as e:
            text = f"(failed to parse excel: {e})"
            ftype = "error"
    elif name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        text = f"[Image: {file.filename}, {len(data)} bytes]"
        ftype = "image"
    else:
        text = data.decode("utf-8", errors="ignore")[:10000]

    return {
        "name": file.filename,
        "type": ftype,
        "text": text[:15000],
        "char_count": len(text),
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint that sends stage updates during generation."""
    import asyncio

    prompt = req.message
    if req.file_context:
        prompt = f"{req.file_context}\n\n{prompt}"

    mode = req.mode
    sys_prompt = get_system_prompt_for_mode(mode)

    async def generate():
        try:
            mode_label = f" [{mode.upper()}]" if mode else ""
            yield f"data: {json.dumps({'stage': f'Querying data tools...{mode_label}'})}\n\n"

            messages = [
                types.Content(role="user", parts=[types.Part(text=f"{STATS_BLOCK}\n\nUSER REQUEST:\n{prompt}")]),
            ]
            tool_log = []
            text_parts = []

            for round_num in range(4):
                response = gclient.models.generate_content(
                    model=CHAT_MODEL, contents=messages,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt, temperature=1.0, tools=GEMINI_TOOLS,
                    ),
                )
                func_calls = []
                text_parts = []
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            func_calls.append(part.function_call)
                        elif hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)

                if not func_calls:
                    break

                for fc in func_calls:
                    args = dict(fc.args) if fc.args else {}
                    tool_log.append({"tool": fc.name, "args": args})
                    yield f"data: {json.dumps({'stage': f'Calling {fc.name}...'})}\n\n"
                    fn = TOOL_FUNCTIONS.get(fc.name)
                    result = fn(**args) if fn else {"error": f"Unknown: {fc.name}"}
                    result = truncate_result(result)
                    messages.append(response.candidates[0].content)
                    messages.append(types.Content(role="user", parts=[
                        types.Part.from_function_response(name=fc.name, response=result)
                    ]))

            draft = "".join(text_parts)
            critique = None
            final = draft

            # Mode-aware two-pass critique
            if req.two_pass and should_critique(prompt, mode, draft):
                yield f"data: {json.dumps({'stage': 'Pass 2: Critiquing draft...'})}\n\n"

                critique_sys = get_critique_prompt_for_mode(mode or "q1")
                if critique_sys:
                    critique_input = f"DRAFT TO REVIEW:\n\n{draft}\n\nReview this draft against the criteria."
                    cr = gclient.models.generate_content(
                        model=CHAT_MODEL, contents=critique_input,
                        config=types.GenerateContentConfig(system_instruction=critique_sys, temperature=0.3),
                    )
                    critique = cr.text or ""

                    if "REWRITE NEEDED" in critique.upper():
                        yield f"data: {json.dumps({'stage': 'Pass 3: Rewriting based on critique...'})}\n\n"
                        rw = gclient.models.generate_content(
                            model=CHAT_MODEL,
                            contents=f"Draft:\n{draft}\n\nCritique:\n{critique}\n\nRewrite fixing every FAIL. Return ONLY the revised script.",
                            config=types.GenerateContentConfig(system_instruction=sys_prompt, temperature=1.0),
                        )
                        final = rw.text or draft

            # Save chat
            chat_id = req.chat_id or str(uuid.uuid4())
            chat_path = CHATS_DIR / f"{chat_id}.json"
            CHATS_DIR.mkdir(parents=True, exist_ok=True)
            if chat_path.exists():
                chat_data = json.loads(chat_path.read_text())
            else:
                chat_data = {"id": chat_id, "title": "", "messages": [], "project_id": req.project_id}
            chat_data["messages"].append({"role": "user", "content": req.message})
            chat_data["messages"].append({
                "role": "assistant", "content": final,
                "tool_log": tool_log,
                "draft": draft if critique else None,
                "critique": critique,
            })
            if not chat_data["title"]:
                chat_data["title"] = req.message[:50]
            chat_path.write_text(json.dumps(chat_data, default=str))

            yield f"data: {json.dumps({'done': True, 'chat_id': chat_id, 'response': final, 'draft': draft if critique else None, 'critique': critique, 'tool_log': tool_log})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/projects/{project_id}")
async def delete_project_endpoint(project_id: str):
    p = PROJECTS_DIR / f"{project_id}.json"
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.get("/api/shows")
async def list_shows():
    """Return list of available shows from briefs data."""
    briefs = _load_briefs()
    shows = {}
    for b in briefs:
        slug = b.get("show_slug")
        name = b.get("show_name") or slug
        if slug and slug not in shows:
            shows[slug] = name
    return [{"slug": s, "name": n} for s, n in sorted(shows.items(), key=lambda x: x[1])]


@app.get("/api/stats")
async def get_stats():
    assets = _load_assets()
    briefs = _load_briefs()
    return {
        "total_assets": len(assets),
        "total_briefs": len(briefs),
        "total_shows": len(set(b.get("show_slug") for b in briefs if b.get("show_slug"))),
    }
