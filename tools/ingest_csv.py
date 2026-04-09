"""
Ingest the asset testing CSV into clean JSON.
See workflows/01_ingest_csv.md for full spec.

Usage:
    python tools/ingest_csv.py
"""
import json
import re
from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/asset_testing_sheet.csv")
OUT_PATH = Path(".tmp/assets.json")


def _to_float(v, strip_chars="$%,"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    for c in strip_chars:
        s = s.replace(c, "")
    try:
        return float(s)
    except ValueError:
        return None


def _drive_file_id(url):
    """Extract Drive file id from a share URL. Returns None if not found."""
    if not url or pd.isna(url):
        return None
    # matches /file/d/<id>/ and /folders/<id> and ?id=<id>
    m = re.search(r"/(?:file/d|folders|d)/([a-zA-Z0-9_-]{20,})", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]{20,})", url)
    return m.group(1) if m else None


def parse_ad_name(name):
    """
    Pipe-delimited Ad Name -> dict of extracted fields.
    Defensive: missing positions -> None.
    """
    if not isinstance(name, str):
        return {}
    parts = [p.strip() for p in name.split("||")]

    def at(i):
        return parts[i] if i < len(parts) and parts[i] else None

    # last chunk often has "__Square" / "__Vertical" size suffix
    experiment = at(17)
    size = None
    if experiment and "__" in experiment:
        experiment, size = experiment.rsplit("__", 1)

    return {
        "country": at(0),
        "platform": at(1),
        "os": at(2),
        "campaign_type": at(3),
        "genre": at(4),
        "market": at(5),
        "ip": at(6),
        "ip_id": at(7),
        "style": at(8),
        "campaign": at(9),
        "test_date_raw": at(10),
        "length": at(11),
        "ratio": at(12),
        "team": at(13),
        "writer_from_name": at(15),
        "ad_code_from_name": at(16),
        "experiment_note": experiment,
        "size": size,
    }


def _s(v):
    """Coerce any cell value to a clean string or None. Handles NaN/floats."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def normalize_row(row):
    ad_name = _s(row.get("Ad Name")) or ""
    parsed = parse_ad_name(ad_name)

    opening_transcript_url = _s(row.get("opening Transcript"))
    script_transcript_url = _s(row.get("Script transcript"))
    asset_drive_url = _s(row.get("Asset Drive Link"))

    fb_growth_raw = _s(row.get("Uploaded On Fb Growth"))
    fb_growth_lower = (fb_growth_raw or "").lower()
    is_active = bool(fb_growth_raw) and fb_growth_lower not in {"not active", "nan", "none"}

    asset = {
        # identity
        "sr_no": int(row["Sr. No."]) if not pd.isna(row.get("Sr. No.")) else None,
        "ad_name_raw": ad_name,
        "ad_code": _s(row.get("Ad code")),
        "opening_code": _s(row.get("Opening \nCode")),
        "script_name": _s(row.get("Script Name")),
        "writer": _s(row.get("Writer")),
        # content
        "opening": _s(row.get("Opening")),
        # parsed from ad name
        **parsed,
        # performance (normalized)
        "opening_lowest_cpi": _to_float(row.get("Opening's \nLowest \nCPI")),
        "total_spend": _to_float(row.get("Total Spends on Testing")),
        "cpi": _to_float(row.get("Actual CPI")),
        "cpm": _to_float(row.get("CPM")),
        "three_sec_play": _to_float(row.get("3Sec Play")),
        "thruplays": _to_float(row.get("thruplays")),
        "video_0_25": _to_float(row.get("Video - 0% - 25%")),
        "video_25_50": _to_float(row.get("Video - 25% - 50%")),
        "video_50_75": _to_float(row.get("Video - 50% - 75%")),
        "video_75_95": _to_float(row.get("Video - 75% - 95%")),
        "video_0_95": _to_float(row.get("Video - 0% - 95%")),
        "completion_impression": _to_float(row.get("Completion/\nImpression")),
        "ctr_cti": _to_float(row.get("CTR*CTI")),
        # links
        "asset_drive_url": asset_drive_url,
        "opening_transcript_url": opening_transcript_url,
        "script_transcript_url": script_transcript_url,
        "opening_transcript_file_id": _drive_file_id(opening_transcript_url),
        "script_transcript_file_id": _drive_file_id(script_transcript_url),
        "upload_date_testing": _s(row.get("Adset Upload Date in Testing")),
        "fb_growth_status": fb_growth_raw,
        "is_active_growth": is_active,
    }
    return asset


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    assets = []
    for _, row in df.iterrows():
        # Skip rows with no opening AND no ad code — probably empty
        if not (row.get("Opening") or row.get("Ad code")):
            continue
        assets.append(normalize_row(row))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(assets, f, indent=2, default=str)

    # Quick stats
    with_cpi = sum(1 for a in assets if a["cpi"] is not None)
    with_opening_doc = sum(1 for a in assets if a["opening_transcript_file_id"])
    with_script_doc = sum(1 for a in assets if a["script_transcript_file_id"])
    unique_opening_docs = len({a["opening_transcript_file_id"] for a in assets if a["opening_transcript_file_id"]})
    unique_script_docs = len({a["script_transcript_file_id"] for a in assets if a["script_transcript_file_id"]})

    print(f"Wrote {len(assets)} assets -> {OUT_PATH}")
    print(f"  with CPI:              {with_cpi}")
    print(f"  with opening doc link: {with_opening_doc}  ({unique_opening_docs} unique)")
    print(f"  with script doc link:  {with_script_doc}  ({unique_script_docs} unique)")


if __name__ == "__main__":
    main()
