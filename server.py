"""
FastAPI backend for Superbot.
Wraps existing data_tools + OpenRouter (multi-model) function calling.
Serves the static frontend + handles chat API.
"""
import json
import os
import sys
import uuid
import time
from datetime import datetime
from pathlib import Path

import httpx
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

# ── Config ──
DEFAULT_MODEL = "google/gemini-2.5-flash"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CHATS_DIR = Path(".tmp/chats")
PROJECTS_DIR = Path(".tmp/projects")
FEEDBACK_PATH = Path(".tmp/feedback.json")

# ── GCS-backed persistence (for Cloud Run) ──
GCS_BUCKET = os.getenv("GCS_BUCKET")
_gcs_client = None
_gcs_bucket = None


def _get_gcs():
    global _gcs_client, _gcs_bucket
    if not GCS_BUCKET:
        return None
    if _gcs_bucket is None:
        try:
            from google.cloud import storage as gcs_storage
            _gcs_client = gcs_storage.Client()
            _gcs_bucket = _gcs_client.bucket(GCS_BUCKET)
        except Exception:
            return None
    return _gcs_bucket


def store_json(path: str, data):
    local = Path(path)
    local.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, default=str)
    local.write_text(content)
    bucket = _get_gcs()
    if bucket:
        try:
            bucket.blob(path).upload_from_string(content, content_type="application/json")
        except Exception:
            pass


def load_json(path: str):
    local = Path(path)
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            pass
    bucket = _get_gcs()
    if bucket:
        try:
            blob = bucket.blob(path)
            if blob.exists():
                content = blob.download_as_text()
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(content)
                return json.loads(content)
        except Exception:
            pass
    return None


def delete_json(path: str):
    local = Path(path)
    if local.exists():
        local.unlink()
    bucket = _get_gcs()
    if bucket:
        try:
            blob = bucket.blob(path)
            if blob.exists():
                blob.delete()
        except Exception:
            pass


def list_json_keys(prefix: str) -> list:
    keys = set()
    bucket = _get_gcs()
    if bucket:
        try:
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith(".json"):
                    keys.add(blob.name)
        except Exception:
            pass
    local_dir = Path(prefix)
    if local_dir.exists():
        for p in local_dir.glob("*.json"):
            keys.add(str(p))
    return sorted(keys, reverse=True)


# ── Load system prompt from chatbot_app.py ──
def _extract_system_prompt():
    src = (Path(__file__).parent / "tools" / "chatbot_app.py").read_text()
    start = src.find('SYSTEM_PROMPT = """') + len('SYSTEM_PROMPT = """')
    end = src.find('"""', start)
    return src[start:end]

_BASE_SYSTEM_PROMPT = _extract_system_prompt()

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

# ── OpenRouter client ──
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set — get one at https://openrouter.ai/keys")


def openrouter_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://superbot.pocketfm.com",
        "X-Title": "Superbot",
    }


def openrouter_chat(model, messages, system=None, temperature=1.0, tools=None):
    """Call OpenRouter chat completions API. Returns the response dict."""
    msgs = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    payload = {"model": model, "messages": msgs, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    resp = httpx.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers=openrouter_headers(),
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


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
MODE: OPENING (180-second hook content)
- Write a ~180-second (650-800 words) opening that instantly hooks viewers.
- May or may NOT be related to the actual story — bizarre/extreme cold opens work.
- CPI = CPM / (CTR × CTI). The opening's job is to MAXIMIZE CTR.
- Structure: HOOK (0-3s) → ESCALATION (3-30s) → TENSION PEAK (30-90s) → MERGE POINT (90-120s) → STORY ENTRY (120-180s)
- Always mark: **[MERGE POINT — transition to base story]**
- The merge must feel seamless. Opening's emotional thread must carry through.
- Cliffhanger alignment: opening's emotional promise must connect to the ending.
""",
    "q1": """
MODE: Q1 SCRIPT (First Quarter — ~2 minutes, 450-530 words)
- Write a Q1 that builds CONVICTION and CONNECTION with the female lead.
- 5-BEAT FORMULA: Shock Hook → Tragic Backstory → Escalating Named Abuse → Identity Reveal → Fated Mate + Cliffhanger
- Female protagonist MANDATORY. First-person voice. Named characters. Quotable cruel dialogue.
""",
    "base": """
MODE: BASE SCRIPT (Full standalone 8-12 minute script, ~5,500-6,000 words)
- Q1 (2 min): Hook + conviction building
- Q2 (3 min): Deepen conflict, introduce love interest
- Q3 (3 min): Climax — maximum tension, power reveal
- Q4 (2 min): Resolution tease + massive cliffhanger + CTA
""",
    "merge": """
MODE: SCRIPT MERGE — Combining two scripts into one.
Ask clarifying questions if unclear: which scripts, target length, which opening, primary show, priority.
Identify strongest elements from each. The merge must feel like ONE coherent story.
""",
    "cliffhanger": """
MODE: CLIFFHANGER RE-WRITE — Improve CTR × CTI.
Rewrite the final 30-60 seconds. Must create maximum unresolved tension.
Types: Unresolved threat, Identity bomb, Impossible choice, Betrayal moment, Power surge.
Keep to 3-5 sentences. Specific, not vague. Align with opening hook.
""",
    "super": """
MODE: SUPERAGENT — Full creative suite.
You can analyze data, write openings/Q1s/base scripts, merge scripts, rewrite cliffhangers.
CPI = CPM / (CTR × CTI). Every recommendation grounded in this formula.
If something is unclear, ask before proceeding.
""",
}

CRITIQUE_SYSTEM = """You are a senior creative director at Pocket FM reviewing a draft response.

Your job is simple: read what the user originally asked for, read the draft, and evaluate whether the draft actually delivers what was requested — clearly, accurately, and effectively.

You understand the Pocket FM ad domain deeply:
- CPI = CPM / (CTR × CTI). Lower CPI = better. CTR is driven by hooks/openings. CTI is driven by cliffhangers.
- Openings (~180s) hook viewers and merge into the base story. They need a clear merge point.
- Q1 scripts (~450-530 words) are the first 2 minutes — they build conviction and connection with the female lead.
- Base scripts (8-12 min, ~5,500-6,000 words) are full standalone ads.
- Cliffhangers must create unresolved tension that drives installs.
- The audience is women 18-35. Female protagonists perform best.

HOW TO CRITIQUE:
1. First, understand what the user asked for. Is it a script? A data report? An analysis?
2. Evaluate ONLY against what was asked.
3. For scripts: Is it emotionally compelling? Hook strong? Characters named? Pacing tight? Cliffhanger land?
4. For data/analysis: Is it accurate? Well-structured? Does it answer the question?
5. For any response: Is anything missing? Could it be sharper?

Keep your critique concise and actionable. Focus on what would make the biggest difference.

End with: VERDICT: PASS (good to go) or REWRITE NEEDED (with specific fixes)."""


# ── Tool calling (OpenAI function format) ──

TOOL_FUNCTIONS = {
    "query_assets": query_assets,
    "get_asset_detail": get_asset_detail,
    "get_show_context": get_show_context,
    "get_opening_stats": get_opening_stats,
    "get_writer_stats": get_writer_stats,
    "get_leaderboard": get_leaderboard,
}

OPENAI_TOOLS = [
    {"type": "function", "function": {
        "name": "query_assets",
        "description": "Query ad assets with exact filters. Use for 'show me TAB ads under $2 CPI'.",
        "parameters": {"type": "object", "properties": {
            "ip": {"type": "string", "description": "Show: TAB, TOLR, WBM, AQB, M3VW, etc."},
            "genre": {"type": "string"}, "writer": {"type": "string"},
            "max_cpi": {"type": "number"}, "min_cpi": {"type": "number"},
            "growth_only": {"type": "boolean"},
            "sort_by": {"type": "string", "description": "cpi/cpi_desc/ctr_cti/total_spend"},
            "limit": {"type": "integer"}, "search_text": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_asset_detail",
        "description": "Full detail for a specific ad code — metrics + full script text.",
        "parameters": {"type": "object", "properties": {
            "ad_code": {"type": "string"},
        }, "required": ["ad_code"]},
    }},
    {"type": "function", "function": {
        "name": "get_show_context",
        "description": "Get show's 10HR base story, character canvas, CPI-cracking notes.",
        "parameters": {"type": "object", "properties": {
            "show": {"type": "string"},
            "section": {"type": "string", "description": "all/base_story/cpi_crack/character_canvas"},
            "max_chars": {"type": "integer"},
        }, "required": ["show"]},
    }},
    {"type": "function", "function": {
        "name": "get_opening_stats",
        "description": "Stats for an opening_code (reuse count, avg CPI) or top most-reused openings.",
        "parameters": {"type": "object", "properties": {
            "opening_code": {"type": "string"},
            "top_n_reused": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_writer_stats",
        "description": "Writer portfolio or leaderboard by avg CPI.",
        "parameters": {"type": "object", "properties": {
            "writer": {"type": "string"},
            "top_n": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_leaderboard",
        "description": "Top N assets by metric (cpi/ctr_cti/retention/spend/cpm).",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string"},
            "ip": {"type": "string"}, "genre": {"type": "string"},
            "growth_only": {"type": "boolean"},
            "limit": {"type": "integer"},
        }},
    }},
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
    base = SYSTEM_PROMPT
    if mode and mode in MODE_INSTRUCTIONS:
        base += "\n\n" + MODE_INSTRUCTIONS[mode]
    return base


def should_critique(draft):
    return len(draft) > 200


def execute_tool_call(name, args_str):
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        return fn(**args)
    except Exception as e:
        return {"error": f"Tool {name} failed: {str(e)}"}


def run_generation(prompt, two_pass=True, mode=None, model=None, chat_history=None):
    """Run tool-calling loop via OpenRouter + optional two-pass critique."""
    model = model or DEFAULT_MODEL
    sys_prompt = get_system_prompt_for_mode(mode)

    # Build messages with conversation history
    messages = []
    if chat_history:
        # Include last N message pairs for context (avoid blowing up token limit)
        recent = chat_history[-10:]  # last 5 exchanges
        for m in recent:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"{STATS_BLOCK}\n\nUSER REQUEST:\n{prompt}"})
    tool_log = []
    msg = {}

    # Tool calling loop
    for _ in range(4):
        resp = openrouter_chat(model, messages, system=sys_prompt, temperature=1.0, tools=OPENAI_TOOLS)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            break

        # Append assistant message with tool calls
        messages.append(msg)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"].get("arguments", "{}")
            tool_log.append({"tool": fn_name, "args": json.loads(fn_args) if isinstance(fn_args, str) else fn_args})
            result = execute_tool_call(fn_name, fn_args)
            result = truncate_result(result)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, default=str)})

    draft = msg.get("content") or ""
    critique = None
    final = draft

    # Smart two-pass critique
    if two_pass and should_critique(draft):
        critique_input = (
            f"USER'S ORIGINAL REQUEST:\n{prompt}\n\n"
            f"DRAFT RESPONSE:\n{draft}\n\n"
            f"Review whether this draft delivers what the user asked for."
        )
        cr = openrouter_chat(model, [{"role": "user", "content": critique_input}], system=CRITIQUE_SYSTEM, temperature=0.3)
        critique = (cr.get("choices") or [{}])[0].get("message", {}).get("content") or ""

        if "REWRITE NEEDED" in critique.upper():
            rw = openrouter_chat(model, [{"role": "user", "content": (
                f"The user asked:\n{prompt}\n\n"
                f"You wrote this draft:\n{draft}\n\n"
                f"Your creative director reviewed it:\n{critique}\n\n"
                f"Now rewrite fixing every issue. Keep what works. Return ONLY the improved response."
            )}], system=sys_prompt, temperature=1.0)
            final = (rw.get("choices") or [{}])[0].get("message", {}).get("content") or draft

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
    mode: Optional[str] = None
    model: Optional[str] = None


class CompareRequest(BaseModel):
    message: str
    models: List[str]
    mode: Optional[str] = None
    file_context: Optional[str] = None


class FeedbackRequest(BaseModel):
    chat_id: str
    msg_idx: int
    rating: str
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


# ── Models ──

# Cache models for 1 hour
_models_cache = {"data": None, "ts": 0}


@app.get("/api/models")
async def list_models():
    """Fetch available models from OpenRouter."""
    now = time.time()
    if _models_cache["data"] and now - _models_cache["ts"] < 3600:
        return _models_cache["data"]

    try:
        resp = httpx.get(f"{OPENROUTER_BASE}/models", headers=openrouter_headers(), timeout=15.0)
        resp.raise_for_status()
        all_models = resp.json().get("data", [])
        # Filter to chat models, return id + name + pricing
        models = []
        for m in all_models:
            models.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("context_length"),
                "pricing": m.get("pricing", {}),
            })
        # Sort by name
        models.sort(key=lambda x: x["name"].lower())
        _models_cache["data"] = models
        _models_cache["ts"] = now
        return models
    except Exception as e:
        if _models_cache["data"]:
            return _models_cache["data"]
        raise HTTPException(500, f"Failed to fetch models: {e}")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    prompt = req.message
    if req.file_context:
        prompt = f"{req.file_context}\n\n{prompt}"

    # Load chat history for context
    chat_history = None
    if req.chat_id:
        existing = load_json(f".tmp/chats/{req.chat_id}.json")
        if existing:
            chat_history = existing.get("messages", [])

    try:
        result = run_generation(prompt, two_pass=req.two_pass, mode=req.mode, model=req.model, chat_history=chat_history)
    except Exception as e:
        raise HTTPException(500, str(e))

    chat_id = req.chat_id or str(uuid.uuid4())
    chat_key = f".tmp/chats/{chat_id}.json"
    chat_data = load_json(chat_key) or {"id": chat_id, "title": "", "messages": [], "project_id": req.project_id}
    chat_data["messages"].append({"role": "user", "content": req.message})
    chat_data["messages"].append({
        "role": "assistant", "content": result["response"],
        "tool_log": result["tool_log"],
        "draft": result.get("draft"), "critique": result.get("critique"),
        "model": req.model or DEFAULT_MODEL,
    })
    if not chat_data["title"]:
        chat_data["title"] = req.message[:50]
    store_json(chat_key, chat_data)

    return {
        "chat_id": chat_id,
        "response": result["response"],
        "draft": result.get("draft"),
        "critique": result.get("critique"),
        "tool_log": result["tool_log"],
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint that sends stage updates during generation."""
    prompt = req.message
    if req.file_context:
        prompt = f"{req.file_context}\n\n{prompt}"

    mode = req.mode
    model = req.model or DEFAULT_MODEL
    sys_prompt = get_system_prompt_for_mode(mode)

    # Load chat history for context
    chat_history = None
    if req.chat_id:
        existing = load_json(f".tmp/chats/{req.chat_id}.json")
        if existing:
            chat_history = existing.get("messages", [])

    async def generate():
        try:
            mode_label = f" [{mode.upper()}]" if mode else ""
            model_short = model.split("/")[-1] if "/" in model else model
            yield f"data: {json.dumps({'stage': f'Using {model_short}...{mode_label}'})}\n\n"

            # Build messages with conversation history
            messages = []
            if chat_history:
                recent = chat_history[-10:]
                for m in recent:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": f"{STATS_BLOCK}\n\nUSER REQUEST:\n{prompt}"})
            tool_log = []
            msg = {}

            for round_num in range(4):
                yield f"data: {json.dumps({'stage': f'Querying data tools... (round {round_num+1})'})}\n\n"
                resp = openrouter_chat(model, messages, system=sys_prompt, temperature=1.0, tools=OPENAI_TOOLS)
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    break

                messages.append(msg)
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args = tc["function"].get("arguments", "{}")
                    tool_log.append({"tool": fn_name, "args": json.loads(fn_args) if isinstance(fn_args, str) else fn_args})
                    yield f"data: {json.dumps({'stage': f'Calling {fn_name}...'})}\n\n"
                    result = execute_tool_call(fn_name, fn_args)
                    result = truncate_result(result)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, default=str)})

            draft = msg.get("content") or ""
            critique = None
            final = draft

            # Smart two-pass critique
            if req.two_pass and should_critique(draft):
                yield f"data: {json.dumps({'stage': 'Pass 2: Critiquing draft...'})}\n\n"
                critique_input = (
                    f"USER'S ORIGINAL REQUEST:\n{prompt}\n\n"
                    f"DRAFT RESPONSE:\n{draft}\n\n"
                    f"Review whether this draft delivers what the user asked for."
                )
                cr = openrouter_chat(model, [{"role": "user", "content": critique_input}], system=CRITIQUE_SYSTEM, temperature=0.3)
                critique = (cr.get("choices") or [{}])[0].get("message", {}).get("content") or ""

                if "REWRITE NEEDED" in critique.upper():
                    yield f"data: {json.dumps({'stage': 'Pass 3: Rewriting based on critique...'})}\n\n"
                    rw = openrouter_chat(model, [{"role": "user", "content": (
                        f"The user asked:\n{prompt}\n\n"
                        f"You wrote this draft:\n{draft}\n\n"
                        f"Your creative director reviewed it:\n{critique}\n\n"
                        f"Now rewrite fixing every issue. Keep what works. Return ONLY the improved response."
                    )}], system=sys_prompt, temperature=1.0)
                    final = (rw.get("choices") or [{}])[0].get("message", {}).get("content") or draft

            # Save chat
            chat_id = req.chat_id or str(uuid.uuid4())
            chat_key = f".tmp/chats/{chat_id}.json"
            chat_data = load_json(chat_key) or {"id": chat_id, "title": "", "messages": [], "project_id": req.project_id}
            chat_data["messages"].append({"role": "user", "content": req.message})
            chat_data["messages"].append({
                "role": "assistant", "content": final,
                "tool_log": tool_log,
                "draft": draft if critique else None,
                "critique": critique,
                "model": model,
            })
            if not chat_data["title"]:
                chat_data["title"] = req.message[:50]
            store_json(chat_key, chat_data)

            yield f"data: {json.dumps({'done': True, 'chat_id': chat_id, 'response': final, 'draft': draft if critique else None, 'critique': critique, 'tool_log': tool_log, 'model': model})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Compare endpoint ──

@app.post("/api/compare")
async def compare_models(req: CompareRequest):
    """Run the same prompt against multiple models and return all results."""
    prompt = req.message
    if req.file_context:
        prompt = f"{req.file_context}\n\n{prompt}"

    async def generate():
        results = {}
        for model_id in req.models[:4]:  # Max 4 models
            model_short = model_id.split("/")[-1] if "/" in model_id else model_id
            yield f"data: {json.dumps({'stage': f'Running {model_short}...'})}\n\n"
            try:
                result = run_generation(prompt, two_pass=False, mode=req.mode, model=model_id)
                results[model_id] = {
                    "response": result["response"],
                    "tool_log": result["tool_log"],
                    "model": model_id,
                }
            except Exception as e:
                results[model_id] = {"response": f"Error: {str(e)}", "tool_log": [], "model": model_id}

        yield f"data: {json.dumps({'done': True, 'results': results})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── CRUD endpoints ──

@app.get("/api/chats")
async def list_chats(project_id: Optional[str] = None):
    chats = []
    for key in list_json_keys(".tmp/chats"):
        try:
            c = load_json(key)
            if not c:
                continue
            if project_id and c.get("project_id") != project_id:
                continue
            chats.append({"id": c["id"], "title": c.get("title", ""), "project_id": c.get("project_id"), "msg_count": len(c.get("messages", []))})
        except Exception:
            pass
    return chats[:50]


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    c = load_json(f".tmp/chats/{chat_id}.json")
    if not c:
        raise HTTPException(404, "Chat not found")
    return c


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    delete_json(f".tmp/chats/{chat_id}.json")
    return {"ok": True}


@app.get("/api/projects")
async def list_projects():
    projects = []
    for key in list_json_keys(".tmp/projects"):
        try:
            p = load_json(key)
            if p:
                projects.append(p)
        except Exception:
            pass
    return projects


@app.post("/api/projects")
async def create_project(req: ProjectRequest):
    pid = datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(2).hex()
    proj = {"id": pid, "name": req.name}
    store_json(f".tmp/projects/{pid}.json", proj)
    return proj


@app.delete("/api/projects/{project_id}")
async def delete_project_endpoint(project_id: str):
    delete_json(f".tmp/projects/{project_id}.json")
    return {"ok": True}


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    fb_key = ".tmp/feedback.json"
    fb = load_json(fb_key) or []
    fb.append({
        "chat_id": req.chat_id, "msg_idx": req.msg_idx,
        "rating": req.rating, "actual_cpi": req.actual_cpi,
        "timestamp": datetime.now().isoformat(),
    })
    store_json(fb_key, fb)
    return {"ok": True}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
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


@app.get("/api/shows")
async def list_shows():
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
