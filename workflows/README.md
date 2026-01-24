# Workflows

This directory contains markdown SOPs (Standard Operating Procedures) that define tasks and how to execute them.

## Workflow Template

Each workflow should include:

1. **Objective** - What this workflow accomplishes
2. **Required Inputs** - What information/data is needed
3. **Tools Used** - Which scripts in `tools/` to execute
4. **Steps** - Detailed procedure in order
5. **Expected Outputs** - What gets delivered and where
6. **Edge Cases** - Known issues and how to handle them
7. **Lessons Learned** - Discoveries from past executions

## Example Workflow Structure

```markdown
# Task Name

## Objective
[Clear statement of what this accomplishes]

## Required Inputs
- Input 1: [description]
- Input 2: [description]

## Tools Used
- `tools/script_name.py`

## Steps
1. Step one
2. Step two
3. Step three

## Expected Outputs
- Output location: [Google Sheets URL / file path]
- Format: [description]

## Edge Cases
- **Issue**: [description]
  - **Solution**: [how to handle]

## Lessons Learned
- [Date]: [discovery or improvement]
```

## Creating New Workflows

Don't create or overwrite workflows without user approval. These are the agent's operating instructions and should be preserved and refined over time.
