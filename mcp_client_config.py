# mcp_client_config.py — add at the very top
import os

from dotenv import load_dotenv

load_dotenv()  # ensures GITHUB_PAT is available regardless of import order elsewhere

from langchain_mcp_adapters.client import MultiServerMCPClient

_servers = {
    "recruitment": {
        "transport": "streamable_http",
        "url": "https://recruitment-mcp-server-1033061769276.us-central1.run.app/mcp",
    },
}

github_pat = os.environ.get(
    "GITHUB_PAT"
)  # .get(), not [...] — degrades gracefully instead of crashing
if github_pat:
    _servers["github"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": github_pat},
    }
else:
    print(
        "ℹ️ GITHUB_PAT not set — GitHub enrichment will be skipped, recruitment tools still work"
    )

mcp_client = MultiServerMCPClient(_servers)
