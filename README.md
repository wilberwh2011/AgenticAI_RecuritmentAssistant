# AI Recruitment Assistant
### Multi-Agent AI System | LangGraph + Vertex AI + Gemini

A production-style autonomous AI recruitment pipeline demonstrating 
enterprise agentic AI architecture using Google Cloud Platform.

---

## What It Does

Automates candidate screening through a 3-agent pipeline:

- **Agent 1 — Retriever**: Semantically searches resumes using
  RAG (Retrieval-Augmented Generation) and Vertex AI embeddings
- **Agent 2 — Evaluator**: Scores and ranks candidates against
  a job description using Gemini 2.5 Flash
- **Agent 3 — Summarizer**: Produces a professional shortlist
  report with interview recommendations

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
├── resumes/           # Candidate resume files (.txt or .pdf)
├── requirements.txt   # Python dependencies
├── .gitignore         # Files excluded from version control
└── .env               # GCP credentials (never committed)
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

Add candidate resumes as `.txt` or `.pdf` files to the `/resumes` folder.

**5 — Run**
```bash
py main.py
```

---

## Architecture
User Query + Job Description

│

▼

┌─────────────────────┐

│   Agent 1           │

│   RAG Retriever     │──── Vertex AI text-embedding-004

│   (Semantic Search) │──── ChromaDB Vector Store

└────────┬────────────┘

│ retrieved candidates

▼

┌─────────────────────┐

│   Agent 2           │

│   Evaluator         │──── Gemini 2.5 Flash (temp=0)

│   (LLM Scoring)     │──── Relative ranking rubric

└────────┬────────────┘

│ scored + ranked candidates

▼

┌─────────────────────┐

│   Agent 3           │

│   Summarizer        │──── Gemini 2.5 Flash (temp=0.2)

│   (Report Writer)   │──── Professional report

└────────┬────────────┘

│

▼

📋 Final Report + saved report_[datetime].txt

---

## Supported Resume Formats

- `.txt` — plain text resumes
- `.pdf` — PDF resumes (multi-page supported)

---

## Demo Script (10 minutes)

1. Show `/resumes` folder with candidate files
2. Choose **[1]** — watch RAG indexing live with embeddings
3. Choose **[2]** — paste a job description, enter search query
4. Watch 3 agents fire in sequence in terminal
5. Show final ranked shortlist report
6. Open saved `report_*.txt` file

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Agentic AI | 3 autonomous agents with defined roles |
| RAG Architecture | Semantic search over private documents |
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