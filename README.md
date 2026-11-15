# AgenticAI Recruitment Assistant
### Multi-Agent AI System | LangGraph · Vertex AI (Gemini) · RAG · MCP · Cloud Run

A prototype multi-agent system that screens candidates against a job description.
Four specialized agents work together in a LangGraph workflow, using retrieval-augmented
generation (RAG) over resumes, structured tool calling, and the Model Context Protocol (MCP).
The project uses MCP in two ways: it calls the GitHub MCP server as a client, and it runs
its own MCP server that exposes recruitment tools to other clients.

The project was built step by step as hands-on learning, with each component added and
refactored in turn (earlier iterations are kept in the repo for reference).

---

## What It Does

Given a job description and a search query, the pipeline:

1. **Agent 1 — Retriever**
   Gemini calls the `search_resumes` tool to run a semantic search over all indexed resumes
   (ChromaDB + Vertex AI embeddings). If the model doesn't make the tool call, the code runs
   the search directly as a fallback.
2. **Agent 2 — Evaluator**
   Looks up each candidate's public GitHub repositories through the **GitHub MCP server**, then
   scores **all candidates together in a single call** (temperature 0), so scores are
   relative to each other rather than judged in isolation.
3. **Agent 3 — Summarizer**
   Produces a ranked shortlist report with recommendations and suggested interview questions.
   If no candidates were found, it stops before calling the model, so it can't produce a
   made-up report.
4. **Agent 4 — Delivery**
   Interprets the user's free-text preference ("show it on screen" / "email it to
   jane@company.com") and calls one of two tools: `send_shortlist_email` or
   `display_report_on_screen`. Any email address the model extracts is re-checked with a
   regex against the user's original input before anything is sent; if the check fails, the
   report is shown on screen instead.

Each agent's behaviour is defined in its own **skill** file (`skills/<name>/SKILL.md`), loaded
as the agent's system prompt.

---

## Architecture

```
            Resumes (.txt / .pdf)
                    │
                    ▼
        ┌───────────────────────┐
        │  RAG Indexer          │── Vertex AI text-embedding-004
        │  (rag_engine.py)      │── ChromaDB vector store
        └───────────┬───────────┘
                    │
     Job description + search query (CLI or POST /run)
                    │
                    ▼
   ┌──────────────────────────────────────────────────┐
   │              LangGraph StateGraph                │
   │                                                  │
   │  Retriever ──► Evaluator ──► Summarizer ──► Delivery
   │   (tool call)   │  (batch      (report)     (tool choice:
   │                 │   scoring)                 email / screen)
   │                 ▼                                │
   │        GitHub MCP server (stdio)                 │
   │        search_repositories                       │
   └──────────────────────────────────────────────────┘
                    │
                    ▼
          Ranked shortlist report

   Separate service: Recruitment MCP server (FastMCP, served over HTTP)
     tools: search_resumes_mcp · evaluate_candidates_mcp · generate_shortlist
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph (async `StateGraph`) |
| LLM | Gemini 2.5 Flash via Vertex AI (LangChain `ChatVertexAI`) |
| Embeddings | Vertex AI `text-embedding-004` |
| Vector database | ChromaDB |
| Tool calling | LangChain `@tool` + `bind_tools` |
| MCP client | `langchain-mcp-adapters` (`MultiServerMCPClient`) → GitHub MCP server |
| MCP server | FastMCP, streamable HTTP transport |
| API | FastAPI + Uvicorn |
| Deployment | Docker → Google Cloud Run (two services) |
| CI/CD | GitHub Actions, Artifact Registry, Cloud Build |
| Language | Python 3.13 |

---

## Project Structure

```
├── main.py                  # FastAPI app (Cloud Run) + interactive CLI
├── agent_graph.py           # LangGraph pipeline: 4 agents + shared RecruitState
├── tools.py                 # Tool functions: search, GitHub check, batch scoring, email, screen
├── rag_engine.py            # Resume loading (TXT/PDF), chunking, embeddings, ChromaDB search
├── skills_loader.py         # Loads and caches SKILL.md files as agent system prompts
├── mcp_client_config.py     # MCP client setup (recruitment server + GitHub MCP server)
├── mcp_server/
│   └── server.py            # Recruitment MCP server (FastMCP)
├── skills/
│   ├── candidate_search/SKILL.md
│   ├── candidate_scoring/SKILL.md
│   ├── shortlist_reporting/SKILL.md
│   └── result_delivery/SKILL.md
├── resumes/                 # Sample resumes (.txt / .pdf)
├── chroma_db/               # Prebuilt vector index (bundled into the container image)
├── Dockerfile               # Agent API image
├── Dockerfile.mcp           # MCP server image
├── cloudbuild.mcp.yaml      # Cloud Build config for the MCP server image
├── .github/workflows/
│   └── deploy.yml           # CI/CD: test → build → deploy app → deploy MCP server → smoke test
└── *_ori.py, *_without*.py  # Earlier iterations (before tool calling / MCP), kept for reference
```

---

## Setup (local)

**1. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment variables** in a `.env` file (never committed):
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1

# Optional: enables GitHub enrichment via the GitHub MCP server (requires Node.js / npx)
GITHUB_PAT=your-github-personal-access-token

# Optional: only needed for the email delivery option
DEMO_EMAIL_SENDER=you@gmail.com
DEMO_EMAIL_APP_PASSWORD=your-gmail-app-password
```

**3. Authenticate with Google Cloud**
```bash
gcloud auth application-default login
```

**4. Add resumes** — drop any number of `.txt` or `.pdf` files into `/resumes`.
No code changes needed; GitHub usernames found in resumes are stored as metadata.

**5. Run the CLI**
```bash
python main.py
```
Menu options: rebuild the resume index, run the full pipeline, or search only.

**Or run the API locally**
```bash
uvicorn main:app --port 8080
```

---

## API Endpoints (Cloud Run)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/run` | Full pipeline. Body: `{"job_description": "...", "query": "..."}`. In API mode, the Delivery agent returns the report instead of waiting for user input. |
| POST | `/search` | Semantic search only. Body: `{"query": "..."}` |

The LangGraph graph (and the GitHub MCP subprocess) is built once at container startup and
reused across requests.

---

## MCP

**As a client:** `mcp_client_config.py` connects to the official GitHub MCP server
(`@modelcontextprotocol/server-github`, stdio transport). The Evaluator uses its
`search_repositories` tool to add each candidate's public repositories to their profile.
If `GITHUB_PAT` is not set, this step is skipped and the pipeline still runs.

**As a server:** `mcp_server/server.py` exposes the project's own capabilities as MCP tools,
deployed as a separate Cloud Run service:

| Tool | Purpose |
|------|---------|
| `search_resumes_mcp` | Semantic resume search |
| `evaluate_candidates_mcp` | Batch-score candidates against a job description |
| `generate_shortlist` | End-to-end search → score → ranked shortlist |

---

## CI/CD (GitHub Actions)

`deploy.yml` runs on every push to `main`:

1. **Test** — installs dependencies and checks that the app imports cleanly
2. **Build & deploy agent API** — Docker build, push to Artifact Registry, deploy to Cloud Run
3. **Build & deploy MCP server** — Cloud Build image, deploy to Cloud Run
4. **Smoke test** — lists the deployed MCP server's tools with MCP Inspector

Pull requests to `main` run the test job only.

---

## Reliability, Evaluation & Observability

Developed on the
[`feat/reliability-observability`](https://github.com/wilberwh2011/AgenticAI_RecuritmentAssistant/tree/feat/reliability-observability) branch:

- **Eval harness** (`eval/`) — four suites with fixture data and pass thresholds:
  retrieval relevance, scoring consistency, summarizer checks, and delivery routing.
  A separate `eval.yml` workflow runs the fast suites on pull requests; the retriever suite
  runs on demand (or nightly, if scheduling is enabled).
- **Tracing** — Langfuse (self-hosted via Docker Compose) plus OpenTelemetry, sending the same
  spans to both Langfuse and Google Cloud Trace.
- **Metrics** — custom retrieval metrics (latency, call count, result count) exported to
  Google Cloud Monitoring.
- **Structured logging** — JSON logs (structlog) where every line carries `trace_id`,
  `session_id` and `agent_node`, so one request can be followed across agents in Cloud Logging.

---

## Design Notes

- **Retrieval and reasoning are separate.** The RAG layer does fast, broad retrieval; the
  agents do slower, explainable reasoning. Each can be tuned or swapped on its own.
- **Batch scoring on purpose.** Candidates are scored together in one call so the ranking is
  comparative; scoring them one at a time would lose that.
- **Deterministic where it matters.** GitHub lookup is triggered by data being present,
  not left to the model's choice; scoring runs at temperature 0.
- **Don't trust LLM output blindly.** Fallbacks cover missed tool calls and unparseable
  scores, and email recipients are validated before anything is sent.

## Known Limitations (prototype)

- The Cloud Run services allow unauthenticated access, and the MCP server accepts any
  Host/Origin. This is fine for a demo but not for production.
- The resume index is bundled into the image, so adding resumes in the cloud means rebuilding
  and redeploying.
- The email option uses Gmail SMTP with an app password; a production version would use a
  managed email service and secret storage.
