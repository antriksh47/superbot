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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

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

SYSTEM_PROMPT = _extract_system_prompt()

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


def run_generation(prompt, two_pass=True):
    """Run tool-calling loop + optional two-pass critique."""
    messages = [
        types.Content(role="user", parts=[types.Part(text=f"{STATS_BLOCK}\n\nUSER REQUEST:\n{prompt}")]),
    ]
    tool_log = []

    # Tool calling loop
    for _ in range(4):
        response = gclient.models.generate_content(
            model=CHAT_MODEL, contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=1.0, tools=GEMINI_TOOLS,
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

    # Two-pass critique for generation requests
    gen_keywords = ["write", "generate", "create", "give me", "q1", "opening", "hook", "script", "draft"]
    is_generation = any(kw in prompt.lower() for kw in gen_keywords)

    if two_pass and is_generation and len(draft) > 200:
        critique_prompt = f"""Review this draft Q1 script. For each criterion score PASS or FAIL:
1. SHOCK HOOK: First sentence under 15 words, visceral image?
2. TRAGIC BACKSTORY: Protagonist at lowest status, specific conditions?
3. NAMED ABUSE: Antagonists named, specific cruelties, quotable dialogue, 3+ escalations?
4. IDENTITY REVEAL: Supernatural power moment, mysterious?
5. FATED MATE + CLIFFHANGER: Unresolved encounter?
6. WORD COUNT: 430-550 words?
7. FEMALE PROTAGONIST?
8. VOICE: First-person/intimate, not literary?
9. PACING: Every beat 3-4 sentences max?
10. DIALOGUE: Antagonist lines cruel enough to screenshot?

DRAFT:\n{draft}\n\nEnd with VERDICT: PASS or REWRITE NEEDED with fixes."""

        cr = gclient.models.generate_content(
            model=CHAT_MODEL, contents=critique_prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        critique = cr.text or ""

        if "REWRITE NEEDED" in critique.upper():
            rw = gclient.models.generate_content(
                model=CHAT_MODEL,
                contents=f"Draft:\n{draft}\n\nCritique:\n{critique}\n\nRewrite fixing every FAIL. Return ONLY the revised script.",
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=1.0),
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


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    project_id: Optional[str] = None
    two_pass: bool = True
    file_context: Optional[str] = None


class FeedbackRequest(BaseModel):
    chat_id: str
    msg_idx: int
    rating: str  # "up", "down", "cpi_logged"
    actual_cpi: Optional[float] = None


class ProjectRequest(BaseModel):
    name: str


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    prompt = req.message
    if req.file_context:
        prompt = f"{req.file_context}\n\n{prompt}"
    try:
        result = run_generation(prompt, two_pass=req.two_pass)
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


@app.get("/api/stats")
async def get_stats():
    assets = _load_assets()
    briefs = _load_briefs()
    return {
        "total_assets": len(assets),
        "total_briefs": len(briefs),
        "total_shows": len(set(b.get("show_slug") for b in briefs if b.get("show_slug"))),
    }
