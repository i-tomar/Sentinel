"""Read-only MCP servers over the real Notion, Jira Cloud and Gmail APIs.

They expose exactly the read tools of the fixture servers (same names, arguments and JSON
shapes), so the agent and ingestion code paths are identical in fixture and live mode.
No write tool exists here.
"""
