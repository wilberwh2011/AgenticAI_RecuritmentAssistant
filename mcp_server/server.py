import os

from fastmcp import FastMCP
from langchain_google_vertexai import ChatVertexAI

from tools import evaluate_candidates_batch, search_resumes

mcp = FastMCP("recruitment-assistant")
llm_eval = ChatVertexAI(model_name="gemini-1.5-pro", temperature=0.2)


@mcp.tool()
def search_resumes_mcp(query: str, k: int = 6) -> list:
    """Search indexed resumes for candidates matching a query."""
    return search_resumes.invoke({"query": query, "k": k})


@mcp.tool()
def evaluate_candidates_mcp(candidates: list, job_description: str) -> list:
    """Batch-score a list of {source, content} candidates against a job description."""
    return evaluate_candidates_batch(candidates, job_description, llm_eval)


@mcp.tool()
def generate_shortlist(job_description: str, top_n: int = 5) -> dict:
    """End-to-end: search, enrich, score, and return a ranked shortlist."""
    candidates = search_resumes.invoke({"query": job_description, "k": top_n * 2})
    scored = evaluate_candidates_batch(candidates, job_description, llm_eval)
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]
    return {"job_description": job_description, "shortlist": ranked}


# if __name__ == "__main__":
#     mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

if __name__ == "__main__":
    print("About to call mcp.run()...", flush=True)
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        allowed_hosts=["*"],  # demo: accept any Host header -- not for production use!
        allowed_origins=[
            "*"
        ],  # demo: accept any browser Origin -- not for production use!
    )
    print(
        "mcp.run() returned — this line should NOT print if it's blocking correctly",
        flush=True,
    )
