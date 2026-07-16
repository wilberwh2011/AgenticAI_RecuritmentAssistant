import os

from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient(
    {
        "recruitment": {
            "transport": "streamable_http",
            "url": "https://recruitment-mcp-server-1033061769276.us-central1.run.app/mcp",
        },
        "github": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHHUB_PAT"]},
        },
    }
)
