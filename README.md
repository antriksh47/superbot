# WAT Framework Project

A reliable AI automation system using **Workflows, Agents, Tools** architecture.

## Quick Start

1. **Set up environment**
   ```bash
   cp .env.template .env
   # Edit .env and add your API keys
   ```

2. **Install dependencies** (when you add Python tools)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Directory Structure

```
.tmp/           # Temporary files (scraped data, intermediate exports)
tools/          # Python scripts for deterministic execution
workflows/      # Markdown SOPs defining tasks and procedures
CLAUDE.md       # Agent instructions and framework documentation
```

## How It Works

- **Workflows** define WHAT to do (instructions in plain language)
- **Agents** (Claude) decide HOW to orchestrate the work
- **Tools** execute tasks deterministically (Python scripts)

See [CLAUDE.md](CLAUDE.md) for complete framework documentation.

## Adding New Capabilities

1. Create a workflow in `workflows/` describing the task
2. Build necessary tools in `tools/`
3. Let the agent coordinate execution

## Notes

- All API keys go in `.env` (never commit this file)
- Final deliverables go to cloud services (Google Sheets, Slides, etc.)
- `.tmp/` is for intermediate processing files only
