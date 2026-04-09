# Run the RAG Chatbot (Streamlit)

## Objective
Give the user a chat interface that can write new openings, rewrite drafts, explain what works, and cite specific past tests.

## Inputs
- `.tmp/chroma_db/` — vector index from 03_build_vector_index
- `GOOGLE_API_KEY` env var (Gemini)

## How to run
```
streamlit run tools/chatbot_app.py
```

## Architecture
1. User sends a message
2. If the message is a generation request, run a retrieval query over ChromaDB (top 10 most similar past tests, optionally filtered by genre/IP/style if the user specified them)
3. Build a system prompt that includes:
   - The framework rules (what "good" means: low CPI, high CTR*CTI, high retention)
   - The retrieved past tests with their performance numbers
   - The user's request
4. Call Gemini 2.5 Pro with the prompt
5. Stream the response back
6. Show a collapsible "Sources" panel listing which past assets were retrieved (ad codes + CPI)

## System prompt principles
- The model MUST cite specific ad codes and metrics when making recommendations ("this structure worked in MAY25885 at $1.86 CPI")
- Lower CPI = better. Benchmark: < $2.50 is top-tier, $2.50–$3.50 is good, > $4 is weak.
- Higher CTR*CTI = better (more clicks/installs per impression)
- Higher retention past 75% = better (script holds attention)
- When rewriting a draft, explain WHY the change is data-backed
- Never invent ad codes or metrics. If retrieval returns nothing relevant, say so.

## Chat features
- Text input + send
- Sidebar filters: genre, IP, style, writer, min/max CPI
- "Top performers" button: shows current top 10 by CPI
- Export chat to text file

## Tool
`tools/chatbot_app.py`
