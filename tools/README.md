# Tools

Python scripts for deterministic execution. Each tool should do one thing well.

## Tool Guidelines

1. **Single Responsibility** - Each script handles one specific task
2. **Clear Interface** - Accept inputs via command-line arguments or config
3. **Error Handling** - Fail gracefully with helpful error messages
4. **Environment Variables** - Load secrets from `.env` using `python-dotenv`
5. **Documentation** - Include docstrings and usage examples

## Example Tool Structure

```python
#!/usr/bin/env python3
"""
Tool Name - Brief description

Usage:
    python tool_name.py --arg1 value1 --arg2 value2
"""

import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()
```

## Common Dependencies

Tools typically use:
- `python-dotenv` - Load environment variables
- `requests` - HTTP requests
- `pandas` - Data manipulation
- `google-auth` - Google API authentication

Add dependencies to `requirements.txt` as needed.
