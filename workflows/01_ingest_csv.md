# Ingest Asset Testing CSV

## Objective
Parse the raw asset testing CSV into clean, structured JSON that downstream tools can work with. Normalize messy metric fields and extract rich metadata hidden in the pipe-delimited `Ad Name` column.

## Inputs
- `data/asset_testing_sheet.csv` — raw export from Google Sheets

## Output
- `.tmp/assets.json` — list of asset dicts with normalized fields

## Ad Name format (pipe-delimited)
Example:
```
USA || Meta || Android || Testing - CPI || Werewolf || USA || The Alpha's Bride || 202319 || GenAI-Cinematic-Motion || TAB1.0-GenAI || 20 May 2025 || Long || 64:4 || Growth || || Yashwant Patil || MAY25508 || <experiment desc>__<size>
```
Typical field positions (0-indexed):
- 0: country, 1: platform, 2: OS, 3: campaign_type, 4: genre, 5: market, 6: IP, 7: IP_id, 8: style, 9: campaign, 10: date, 11: length, 12: ratio, 13: team, 14: (empty), 15: writer, 16: ad_code, 17: experiment_note

Not every row has all 18 fields. Be defensive — fall back to `None` for missing.

## Normalization rules
- Strip `$` from CPI/CPM/Spends → float
- Strip `%` from CTR*CTI and retention % fields → float (e.g. "0.60%" → 0.60)
- `NaN` → `None`
- Keep original string too for display (`_raw` suffix)

## Tool
`tools/ingest_csv.py`

## Edge cases learned
- The `Opening Code` and `Ad code` sometimes match, sometimes differ (Opening Code = hook source, Ad code = actual test asset)
- Many rows share the same script transcript URL → dedupe downstream
- Rows where the hook was reused across tests will have the SAME `Opening` text — this is signal, not noise (compare CPIs across reuses)
