# Agentic Contribution Flow (fin-kit)

1. User requests a new function via MCP intake (fruit-thrower/data-mcp).
2. Request spec JSON is generated.
3. Run:

```bash
python scripts/agent_pipeline.py --request <request.json> --push
```

4. Script triages basic viability and appends code to target module path.
5. Commit is created on `agent` branch.
6. Human reviews PR and merges.
7. After merge, fruit-thrower should re-index fin-kit so MCP search includes new function.

## Reindex command (fruit-thrower)

```bash
python main.py --index-dir ./.code_index-fin-kit index --repo /path/to/fin-kit
```
