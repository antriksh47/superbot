"""
Meta Ad Library integration (Phase 2 — STUB).
See workflows/05_meta_ad_library.md for the planned design.

Blocked on: Meta developer access token with `ads_read` scope.

Once the token is available, this module will:
  1. search() — pull active short-drama ads in the US by keyword/industry
  2. by_page() — list all active ads from a competitor page id
  3. extract_hooks() — transcribe first ~5s of video ads into opening text
  4. upsert_to_chroma() — add entries to the same collection with source='ad_library'
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("META_AD_LIBRARY_TOKEN")
BASE_URL = "https://graph.facebook.com/v19.0/ads_archive"

# Competitor pages to track once we have access.
# Fill these in with real Facebook page IDs.
COMPETITOR_PAGE_IDS = {
    # "ReelShort": "",
    # "DramaBox": "",
    # "GoodShort": "",
    # "FlexTV": "",
    # "MoboReels": "",
}


def _require_token():
    if not ACCESS_TOKEN:
        raise RuntimeError(
            "META_AD_LIBRARY_TOKEN not set. Create a Meta developer app with "
            "Ad Library API access and put the token in .env"
        )


def search(query, country="US", limit=100):
    """Search active ads matching a query in a given country."""
    _require_token()
    params = {
        "access_token": ACCESS_TOKEN,
        "search_terms": query,
        "ad_reached_countries": f"['{country}']",
        "ad_active_status": "ACTIVE",
        "ad_type": "ALL",
        "fields": (
            "id,page_id,page_name,ad_creative_bodies,ad_creative_link_captions,"
            "ad_creative_link_descriptions,ad_snapshot_url,ad_delivery_start_time,"
            "impressions,spend,currency,languages,publisher_platforms"
        ),
        "limit": limit,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def by_page(page_id, country="US", limit=100):
    """Pull active ads from a specific competitor page."""
    _require_token()
    params = {
        "access_token": ACCESS_TOKEN,
        "search_page_ids": f"[{page_id}]",
        "ad_reached_countries": f"['{country}']",
        "ad_active_status": "ACTIVE",
        "ad_type": "ALL",
        "fields": (
            "id,page_id,page_name,ad_creative_bodies,ad_snapshot_url,"
            "ad_delivery_start_time,impressions,spend"
        ),
        "limit": limit,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def main():
    print("Meta Ad Library tool — Phase 2 stub.")
    print("Set META_AD_LIBRARY_TOKEN in .env and fill COMPETITOR_PAGE_IDS to enable.")
    if not ACCESS_TOKEN:
        return
    print("Token present. Test fetch for 'werewolf drama':")
    try:
        data = search("werewolf drama", limit=5)
        for ad in data:
            print(f"- {ad.get('page_name')}: {(ad.get('ad_creative_bodies') or [''])[0][:100]}")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    main()
