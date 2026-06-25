# AI Recruitment Assistant
### Multi-Agent AI System | LangGraph + Vertex AI + Gemini

A production-style autonomous AI recruitment pipeline demonstrating
enterprise agentic AI architecture using Google Cloud Platform.

---

## What It Does

Automates candidate screening through a 3-agent pipeline that works
with any number of resumes in `.txt` or `.pdf` format:

- **Agent 1 — Retriever**: Semantically searches all indexed resumes
  using RAG (Retrieval-Augmented Generation) and Vertex AI embeddings
- **Agent 2 — Evaluator**: Dynamically scores and ranks ALL retrieved
  candidates relative to each other against the job description
- **Agent 3 — Summarizer**: Produces a professional shortlist report
  with ranked recommendations and suggested interview questions

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| LLM | Gemini 2.5 Flash via Vertex AI |
| Embeddings | text-embedding-004 via Vertex AI |
| Vector Database | ChromaDB |
| Cloud Platform | Google Cloud Platform (GCP) |
| Language | Python 3.13 |

---

## Project Structure
Demo/

├── main.py            # CLI entry point

├── agent_graph.py     # Multi-agent LangGraph pipeline

├── rag_engine.py      # RAG engine and vector store

├── resumes/           # Add any number of resumes here

│   ├── *.txt          # Plain text resumes

│   └── *.pdf          # PDF resumes (multi-page supported)

├── requirements.txt   # Python dependencies

├── .gitignore         # Files excluded from version control

└── .env               # GCP credentials (never committed)

---

## Adding Resumes

Simply drop any number of `.txt` or `.pdf` resume files into
the `/resumes` folder — no code changes needed. The pipeline
automatically discovers, loads, and indexes all files on rebuild.
resumes/

├── resume_john.txt         ← plain text

├── resume_sarah.txt        ← plain text

├── resume_mike.txt         ← plain text

└── Resume_Alex Chen.pdf    ← PDF, multi-page supported

When you run option **[1]** from the CLI, you will see:
📄 Loading resumes...

Loading TXT: resume_john.txt

Loading TXT: resume_mike.txt

Loading TXT: resume_sarah.txt

Loading PDF: Resume_Alex Chen.pdf

✅ Loaded 4 file(s) → 9 document chunk(s)

---

## Setup

**1 — Create virtual environment**
```bash
py -m venv .venv
.venv\Scripts\activate
```

**2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**3 — Configure environment**

Create a `.env` file:
GOOGLE_CLOUD_PROJECT=your-project-id

GOOGLE_CLOUD_REGION=us-central1

Authenticate with GCP:
```bash
gcloud auth application-default login
```

**4 — Add resumes**

Drop any number of `.txt` or `.pdf` resume files into `/resumes`.
No code changes needed — the system adapts automatically.

**5 — Run**
```bash
py main.py
```

---

## Architecture
     Any number of resumes (.txt or .pdf)
                  │
                  ▼
        ┌─────────────────┐
        │   RAG Indexer   │──── Vertex AI text-embedding-004
        │   (on demand)   │──── ChromaDB Vector Store
        └────────┬────────┘
                 │
     User Query + Job Description
                 │
                 ▼
    ┌────────────────────────┐
    │   Agent 1              │
    │   RAG Retriever        │──── Semantic similarity search
    │   (dynamic candidates) │──── Returns top N matches
    └────────────┬───────────┘
                 │ N candidates (dynamic)
                 ▼
    ┌────────────────────────┐
    │   Agent 2              │
    │   Evaluator            │──── Gemini 2.5 Flash (temp=0)
    │   (scores all N)       │──── Relative ranking rubric
    └────────────┬───────────┘
                 │ scored + ranked
                 ▼
    ┌────────────────────────┐
    │   Agent 3              │
    │   Summarizer           │──── Gemini 2.5 Flash (temp=0.2)
    │   (final report)       │──── Professional shortlist
    └────────────┬───────────┘
                 │
                 ▼
    📋 Final Report + saved report_[datetime].txt

---

## CLI Options
[1] Load new resumes and rebuild index

→ Scans /resumes for all .txt and .pdf files

→ Rebuilds ChromaDB vector store from scratch

→ Run this whenever you add or remove resumes
[2] Run recruitment pipeline

→ Enter any job description interactively

→ Enter a search query

→ All 3 agents run automatically

→ Saves report to report_[datetime].txt
[3] Search candidates only (no evaluation)

→ Raw semantic search across all indexed resumes

→ No LLM evaluation — fast broad discovery
[4] Exit

---

## How the Evaluator Scales

The Evaluator Agent dynamically adapts to however many candidates
the Retriever returns — no hardcoded limits:

- 3 resumes in folder → evaluates 3 candidates
- 10 resumes in folder → evaluates 10 candidates
- 50 resumes in folder → evaluates 50 candidates

All candidates are scored **relative to each other** in a single
LLM call, ensuring consistent and fair comparison regardless of
how many are in the pool.

---

## Supported Resume Formats

| Format | Support | Notes |
|--------|---------|-------|
| `.txt` | ✅ Full | Plain text, any structure |
| `.pdf` | ✅ Full | Multi-page supported |
| Other | ⚠️ Skipped | Logged with warning |

---

## Demo Script (10 minutes)

1. Show `/resumes` folder — mix of `.txt` and `.pdf` files
2. Choose **[1]** — watch all files indexed live with chunk count
3. Choose **[2]** — paste a real job description, enter query
4. Watch 3 agents fire in sequence in terminal
5. Show final ranked shortlist with scores
6. Open saved `report_*.txt` to show professional output

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Agentic AI | 3 autonomous agents with defined roles |
| RAG Architecture | Semantic search over private documents |
| Dynamic Scaling | Works with any number of resumes |
| Multi-Agent Orchestration | LangGraph state machine |
| Cloud-Native AI | GCP Vertex AI + Gemini |
| Enterprise Pattern | Separation of retrieval vs reasoning |
| Production Practices | .gitignore, requirements.txt, .env |

---

## Why This Architecture Matters

The RAG layer handles retrieval at scale — fast and broad.
The agent layer handles reasoning — accurate and explainable.
Separating them allows independent tuning and component swapping
without rebuilding the entire system — a core enterprise AI principle.

Adding new resumes requires zero code changes — just drop files
into `/resumes` and rebuild the index. This mirrors how enterprise
document pipelines work at scale.