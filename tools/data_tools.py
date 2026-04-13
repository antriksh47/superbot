"""
Structured data tools for the Hook Lab chatbot.

These functions provide EXACT, filtered, sorted access to ad assets and show
briefs — no fuzzy embedding search. The Gemini model calls these via function
calling to get precise data before generating responses.

All functions return plain dicts/lists that can be JSON-serialized and injected
into the LLM context.
"""
import json
from collections import defaultdict
from pathlib import Path

# Resolve paths relative to the repo root (one level up from tools/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_PATH = _REPO_ROOT / ".tmp/assets_with_text.json"
ASSETS_BASIC_PATH = _REPO_ROOT / ".tmp/assets.json"
BRIEFS_PATH = _REPO_ROOT / ".tmp/show_briefs.json"

# ── Lazy-loaded singletons ──

_assets = None
_briefs = None


def _load_assets():
    global _assets
    if _assets is not None:
        return _assets
    p = ASSETS_PATH if ASSETS_PATH.exists() else ASSETS_BASIC_PATH
    if not p.exists():
        _assets = []
        return _assets
    _assets = json.loads(p.read_text())
    return _assets


def _load_briefs():
    global _briefs
    if _briefs is not None:
        return _briefs
    if not BRIEFS_PATH.exists():
        _briefs = []
        return _briefs
    _briefs = json.loads(BRIEFS_PATH.read_text())
    return _briefs


# ── Show name mapping ──

SHOW_ALIASES = {
    "tab": "The Alpha's Bride",
    "the alpha's bride": "The Alpha's Bride",
    "tolr": "Twists of Love & Revenge",
    "twists of love": "Twists of Love & Revenge",
    "wbm": "Wolves of Blood Moon",
    "wobm": "Wolves of Blood Moon",
    "wolves of blood moon": "Wolves of Blood Moon",
    "aqb": "A Queen Betrayed",
    "a queen betrayed": "A Queen Betrayed",
    "m3vw": "My Three Vampire Wives",
    "my three vampire wives": "My Three Vampire Wives",
    "tam": "The Alpha's Mark",
    "the alpha's mark": "The Alpha's Mark",
    "c&c": "Crushed & Crowned",
    "crushed": "Crushed & Crowned",
    "bma": "Blood Moon Academy",
    "tdmb": "The Devil's Mark Burns",
}


def _resolve_ip(name):
    """Resolve an abbreviation or partial name to the canonical IP name."""
    if not name:
        return None
    key = name.strip().lower()
    return SHOW_ALIASES.get(key, name)


# =====================================================================
# TOOL 1: query_assets
# =====================================================================

def query_assets(
    ip: str = None,
    genre: str = None,
    writer: str = None,
    style: str = None,
    max_cpi: float = None,
    min_cpi: float = None,
    growth_only: bool = False,
    sort_by: str = "cpi",
    limit: int = 20,
    search_text: str = None,
) -> dict:
    """
    Query ad assets with exact filters and sorting. Returns structured results.

    Args:
        ip: Show name or abbreviation (TAB, TOLR, WBM, etc.)
        genre: Genre filter (Werewolf, Romance, etc.)
        writer: Writer name filter (partial match)
        style: Style filter (partial match)
        max_cpi: Maximum CPI threshold
        min_cpi: Minimum CPI threshold
        growth_only: If True, only return assets promoted to growth
        sort_by: Sort field — "cpi" (asc), "cpi_desc", "ctr_cti" (desc), "total_spend" (desc)
        limit: Max results (default 20, max 50)
        search_text: Text search across opening, script_name, ad_name_raw (partial match)

    Returns:
        Dict with "count", "results" (list of asset summaries), "query_summary"
    """
    assets = _load_assets()
    limit = min(limit, 50)

    resolved_ip = _resolve_ip(ip) if ip else None

    filtered = []
    for a in assets:
        if resolved_ip:
            asset_ip = (a.get("ip") or "").strip()
            if asset_ip.lower() != resolved_ip.lower():
                continue
        if genre:
            if (a.get("genre") or "").lower() != genre.lower():
                continue
        if writer:
            if writer.lower() not in (a.get("writer") or "").lower():
                continue
        if style:
            if style.lower() not in (a.get("style") or "").lower():
                continue
        if max_cpi is not None:
            cpi = a.get("cpi")
            if cpi is None or cpi > max_cpi:
                continue
        if min_cpi is not None:
            cpi = a.get("cpi")
            if cpi is None or cpi < min_cpi:
                continue
        if growth_only:
            if not a.get("is_active_growth"):
                continue
        if search_text:
            haystack = " ".join([
                a.get("opening") or "",
                a.get("script_name") or "",
                a.get("ad_name_raw") or "",
                a.get("ad_code") or "",
            ]).lower()
            if search_text.lower() not in haystack:
                continue
        filtered.append(a)

    # Sort
    if sort_by == "cpi":
        filtered.sort(key=lambda a: a.get("cpi") or 999)
    elif sort_by == "cpi_desc":
        filtered.sort(key=lambda a: -(a.get("cpi") or 0))
    elif sort_by == "ctr_cti":
        filtered.sort(key=lambda a: -(a.get("ctr_cti") or 0))
    elif sort_by == "total_spend":
        filtered.sort(key=lambda a: -(a.get("total_spend") or 0))

    results = []
    for a in filtered[:limit]:
        results.append({
            "ad_code": a.get("ad_code"),
            "opening_code": a.get("opening_code"),
            "opening": a.get("opening"),
            "script_name": a.get("script_name"),
            "ip": a.get("ip"),
            "genre": a.get("genre"),
            "style": a.get("style"),
            "writer": a.get("writer"),
            "cpi": a.get("cpi"),
            "cpm": a.get("cpm"),
            "ctr_cti": a.get("ctr_cti"),
            "total_spend": a.get("total_spend"),
            "is_active_growth": a.get("is_active_growth"),
            "opening_lowest_cpi": a.get("opening_lowest_cpi"),
        })

    return {
        "count": len(filtered),
        "showing": len(results),
        "query_summary": f"ip={resolved_ip}, genre={genre}, writer={writer}, max_cpi={max_cpi}, growth_only={growth_only}, sort={sort_by}",
        "results": results,
    }


# =====================================================================
# TOOL 2: get_asset_detail
# =====================================================================

def get_asset_detail(ad_code: str) -> dict:
    """
    Get the full detail for a specific ad code, including opening text,
    full script text, and all metrics.
    """
    assets = _load_assets()
    for a in assets:
        if (a.get("ad_code") or "").lower() == ad_code.lower():
            return {
                "found": True,
                "ad_code": a.get("ad_code"),
                "opening_code": a.get("opening_code"),
                "opening": a.get("opening"),
                "script_name": a.get("script_name"),
                "writer": a.get("writer"),
                "ip": a.get("ip"),
                "genre": a.get("genre"),
                "style": a.get("style"),
                "cpi": a.get("cpi"),
                "cpm": a.get("cpm"),
                "ctr_cti": a.get("ctr_cti"),
                "total_spend": a.get("total_spend"),
                "is_active_growth": a.get("is_active_growth"),
                "opening_lowest_cpi": a.get("opening_lowest_cpi"),
                "three_sec_play": a.get("three_sec_play"),
                "video_75_95": a.get("video_75_95"),
                "completion_impression": a.get("completion_impression"),
                "opening_text": (a.get("opening_text") or "")[:3000],
                "script_text": (a.get("script_text") or "")[:8000],
            }
    return {"found": False, "ad_code": ad_code, "error": f"No asset found with ad_code '{ad_code}'"}


# =====================================================================
# TOOL 3: get_show_context
# =====================================================================

def get_show_context(show: str, section: str = "all", max_chars: int = 12000) -> dict:
    """
    Get show-level context: base story, CPI-cracking notes, character canvas.

    Args:
        show: Show name or abbreviation (TAB, TOLR, WBM, AQB, etc.)
        section: "all", "base_story", "cpi_crack", "character_canvas", "hub"
        max_chars: Maximum chars to return (default 12000)

    Returns:
        Dict with show info and text content.
    """
    briefs = _load_briefs()
    resolved = _resolve_ip(show)
    slug_prefix = show.strip().lower().replace(" ", "_").replace("'", "")

    # Find matching brief packages
    matches = []
    for b in briefs:
        slug = b.get("show_slug", "")
        name = b.get("name", "")
        if (slug_prefix in slug or
            (resolved and resolved.lower() in name.lower()) or
            show.lower() in slug or
            show.lower() in name.lower()):
            matches.append(b)

    if not matches:
        return {"found": False, "show": show, "resolved": resolved, "error": f"No brief found for '{show}'"}

    # Collect text by type
    content = {"hub": [], "base_story": [], "cpi_crack": [], "character_canvas": [], "other": []}
    for b in matches:
        if b.get("hub_text"):
            content["hub"].append(f"[{b.get('name', '?')}]\n{b['hub_text']}")
        for ld in b.get("linked_docs", []):
            name = ld.get("name", "")
            text = ld.get("text", "")
            if not text:
                continue
            # Classify
            nl = name.lower()
            if "10hr" in nl or "10 hr" in nl or "cms script" in nl:
                content["base_story"].append(f"[{name}]\n{text}")
            elif "rework" in nl or "rewrite" in nl or "crack" in nl:
                content["cpi_crack"].append(f"[{name}]\n{text}")
            elif "canvas" in nl or "_cc" in nl or "cc_" in nl:
                content["character_canvas"].append(f"[{name}]\n{text}")
            else:
                content["other"].append(f"[{name}]\n{text}")

    # Filter by requested section — prioritize story content over hub metadata
    if section == "all":
        sections_to_include = ["base_story", "cpi_crack", "character_canvas", "hub", "other"]
    else:
        sections_to_include = [section]

    output_parts = []
    total_chars = 0
    for s in sections_to_include:
        for text in content.get(s, []):
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 500:
                    output_parts.append(f"=== {s.upper()} ===\n{text[:remaining]}...[truncated]")
                break
            output_parts.append(f"=== {s.upper()} ===\n{text}")
            total_chars += len(text)

    return {
        "found": True,
        "show": show,
        "resolved_name": resolved,
        "matching_briefs": len(matches),
        "sections_available": {k: len(v) for k, v in content.items() if v},
        "total_chars": total_chars,
        "content": "\n\n".join(output_parts),
    }


# =====================================================================
# TOOL 4: get_opening_stats
# =====================================================================

def get_opening_stats(opening_code: str = None, top_n_reused: int = 20) -> dict:
    """
    Get stats for a specific opening_code, or the top N most-reused opening codes.

    If opening_code is provided: returns all ads that used this opening, with avg CPI, min CPI, reuse count.
    If opening_code is None: returns the top N most-reused opening codes with performance stats.
    """
    assets = _load_assets()

    by_opening = defaultdict(list)
    for a in assets:
        oc = a.get("opening_code")
        if oc:
            by_opening[oc].append(a)

    if opening_code:
        ads = by_opening.get(opening_code, [])
        if not ads:
            return {"found": False, "opening_code": opening_code, "error": f"No ads found with opening_code '{opening_code}'"}
        cpis = [a["cpi"] for a in ads if a.get("cpi") is not None]
        return {
            "found": True,
            "opening_code": opening_code,
            "opening_text": ads[0].get("opening"),
            "reuse_count": len(ads),
            "avg_cpi": round(sum(cpis) / len(cpis), 2) if cpis else None,
            "min_cpi": round(min(cpis), 2) if cpis else None,
            "max_cpi": round(max(cpis), 2) if cpis else None,
            "growth_count": sum(1 for a in ads if a.get("is_active_growth")),
            "ips_used": list(set(a.get("ip") or "?" for a in ads)),
            "writers": list(set(a.get("writer") or "?" for a in ads)),
            "ad_codes": [{"ad_code": a.get("ad_code"), "cpi": a.get("cpi"), "growth": a.get("is_active_growth")} for a in sorted(ads, key=lambda x: x.get("cpi") or 999)][:15],
        }

    # Top N most reused
    ranked = sorted(by_opening.items(), key=lambda x: -len(x[1]))[:top_n_reused]
    results = []
    for oc, ads in ranked:
        cpis = [a["cpi"] for a in ads if a.get("cpi") is not None]
        results.append({
            "opening_code": oc,
            "opening_text": (ads[0].get("opening") or "")[:200],
            "reuse_count": len(ads),
            "avg_cpi": round(sum(cpis) / len(cpis), 2) if cpis else None,
            "min_cpi": round(min(cpis), 2) if cpis else None,
            "growth_count": sum(1 for a in ads if a.get("is_active_growth")),
        })
    return {"top_reused_openings": results}


# =====================================================================
# TOOL 5: get_writer_stats
# =====================================================================

def get_writer_stats(writer: str = None, top_n: int = 15) -> dict:
    """
    Get performance stats for a specific writer, or top N writers by avg CPI.

    If writer is provided: full portfolio breakdown.
    If writer is None: leaderboard of top writers.
    """
    assets = _load_assets()

    by_writer = defaultdict(list)
    for a in assets:
        w = a.get("writer")
        if w:
            by_writer[w].append(a)

    if writer:
        # Partial match
        matched_writer = None
        for w in by_writer:
            if writer.lower() in w.lower():
                matched_writer = w
                break
        if not matched_writer:
            return {"found": False, "writer": writer, "error": f"No writer matching '{writer}'"}
        ads = by_writer[matched_writer]
        cpis = [a["cpi"] for a in ads if a.get("cpi") is not None]
        by_ip = defaultdict(list)
        for a in ads:
            by_ip[a.get("ip") or "?"].append(a.get("cpi"))
        return {
            "found": True,
            "writer": matched_writer,
            "total_ads": len(ads),
            "avg_cpi": round(sum(cpis) / len(cpis), 2) if cpis else None,
            "min_cpi": round(min(cpis), 2) if cpis else None,
            "growth_count": sum(1 for a in ads if a.get("is_active_growth")),
            "by_ip": {ip: {"count": len(v), "avg_cpi": round(sum(c for c in v if c) / max(1, len([c for c in v if c])), 2)} for ip, v in by_ip.items()},
            "top_ads": [{"ad_code": a.get("ad_code"), "cpi": a.get("cpi"), "ip": a.get("ip"), "opening": (a.get("opening") or "")[:150]} for a in sorted(ads, key=lambda x: x.get("cpi") or 999)][:10],
        }

    # Leaderboard
    results = []
    for w, ads in by_writer.items():
        cpis = [a["cpi"] for a in ads if a.get("cpi") is not None]
        if len(cpis) >= 5:
            results.append({
                "writer": w,
                "total_ads": len(ads),
                "avg_cpi": round(sum(cpis) / len(cpis), 2),
                "min_cpi": round(min(cpis), 2),
                "growth_count": sum(1 for a in ads if a.get("is_active_growth")),
            })
    results.sort(key=lambda x: x["avg_cpi"])
    return {"writer_leaderboard": results[:top_n]}


# =====================================================================
# TOOL 6: get_leaderboard
# =====================================================================

def get_leaderboard(
    metric: str = "cpi",
    ip: str = None,
    genre: str = None,
    growth_only: bool = False,
    limit: int = 25,
) -> dict:
    """
    Get top N assets by a specific metric.

    Args:
        metric: "cpi" (lowest), "ctr_cti" (highest), "retention" (highest video_75_95),
                "spend" (highest total_spend), "cpm" (lowest)
        ip: Optional show filter
        genre: Optional genre filter
        growth_only: Only growth-promoted assets
        limit: Max results (default 25)
    """
    assets = _load_assets()
    resolved_ip = _resolve_ip(ip) if ip else None

    filtered = []
    for a in assets:
        if resolved_ip and (a.get("ip") or "").lower() != resolved_ip.lower():
            continue
        if genre and (a.get("genre") or "").lower() != genre.lower():
            continue
        if growth_only and not a.get("is_active_growth"):
            continue
        filtered.append(a)

    metric_map = {
        "cpi": ("cpi", False),
        "ctr_cti": ("ctr_cti", True),
        "retention": ("video_75_95", True),
        "spend": ("total_spend", True),
        "cpm": ("cpm", False),
    }
    field, reverse = metric_map.get(metric, ("cpi", False))
    with_metric = [a for a in filtered if a.get(field) is not None and a.get(field, -1) > 0]
    with_metric.sort(key=lambda a: a[field], reverse=reverse)

    results = []
    for a in with_metric[:limit]:
        results.append({
            "ad_code": a.get("ad_code"),
            "opening_code": a.get("opening_code"),
            "opening": (a.get("opening") or "")[:200],
            "script_name": a.get("script_name"),
            "ip": a.get("ip"),
            "writer": a.get("writer"),
            field: a.get(field),
            "cpi": a.get("cpi"),
            "is_active_growth": a.get("is_active_growth"),
        })

    return {
        "metric": metric,
        "field": field,
        "direction": "highest" if reverse else "lowest",
        "total_matching": len(filtered),
        "showing": len(results),
        "results": results,
    }
