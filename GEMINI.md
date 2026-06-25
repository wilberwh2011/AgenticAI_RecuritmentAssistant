# AI Recruitment Assistant — Gemini CLI Context

## Project Overview
This is a multi-agent AI recruitment pipeline built with LangGraph 
and Vertex AI. It screens candidates using 3 autonomous agents.

## How to Run
```bash
py main.py
```

## Key Files
- `main.py` — CLI entry point
- `agent_graph.py` — 3-agent LangGraph pipeline  
- `rag_engine.py` — RAG engine with ChromaDB
- `resumes/` — candidate resume files

## Demo Commands
To rebuild the resume index:
- Choose option [1] in main.py

To run full recruitment pipeline:
- Choose option [2] in main.py
- Enter job description, type END when done
- Enter search query

## Architecture
Agent 1 (Retriever) → Agent 2 (Evaluator) → Agent 3 (Summarizer)