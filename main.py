import os
import sys
from datetime import date
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from rag_engine import build_vector_store, load_vector_store
from agent_graph import build_graph, RecruitState

# ---------------------------------------------------------
# FASTAPI SETUP (for Cloud Run)
# ---------------------------------------------------------

app = FastAPI(title="AI Recruitment Assistant")

class PipelineRequest(BaseModel):
    job_description: str
    query: str

class SearchRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run_pipeline_api(payload: PipelineRequest):
    """
    Cloud Run endpoint:
    POST /run
    {
        "job_description": "...",
        "query": "..."
    }
    """
    graph = build_graph()

    initial_state = RecruitState(
        job_description=payload.job_description,
        query=payload.query,
        retrieved_candidates=[],
        evaluated_candidates=[],
        final_report=""
    )

    result = graph.invoke(initial_state)
    return result


@app.post("/search")
def search_api(payload: SearchRequest):
    """
    Cloud Run endpoint:
    POST /search
    {
        "query": "..."
    }
    """
    vs = load_vector_store()
    from rag_engine import search_candidates
    results = search_candidates(payload.query, vs)
    return {"results": results}


# ---------------------------------------------------------
# ORIGINAL CLI MODE (unchanged)
# ---------------------------------------------------------

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║        🤖 AI Recruitment Assistant                       ║
║        Powered by LangGraph + Vertex AI + Gemini         ║
╚══════════════════════════════════════════════════════════╝
    """)


def print_menu():
    print("""
What would you like to do?
  [1] Load new resumes and rebuild index
  [2] Run recruitment pipeline
  [3] Search candidates only (no evaluation)
  [4] Exit
""")


def get_job_description():
    print("\n📋 Enter the job description.")
    print("(Type END on a new line when done)\n")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def get_search_query():
    print("\n🔍 Enter your search query:")
    print("Example: 'Python GCP AI experience Kubernetes'")
    return input("> ").strip()


def run_pipeline(job_description: str, query: str):
    print("\n🚀 Starting Multi-Agent Recruitment Pipeline...")
    print("=" * 60)

    graph = build_graph()

    initial_state = RecruitState(
        job_description=job_description,
        query=query,
        retrieved_candidates=[],
        evaluated_candidates=[],
        final_report=""
    )

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("📋 FINAL RECRUITMENT REPORT")
    print(f"📅 Generated: {date.today().strftime('%B %d, %Y')}")
    print("=" * 60)
    print(result["final_report"])

    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"report_{date.today().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"RECRUITMENT REPORT — {date.today().strftime('%B %d, %Y')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Job Description:\n{job_description}\n")
        f.write("=" * 60 + "\n")
        f.write(result["final_report"])

    print(f"\n💾 Report saved to: {report_path}")
    return result


def search_only(query: str):
    from rag_engine import search_candidates
    print("\n🔍 Running search only — no evaluation...")
    vs = load_vector_store()
    search_candidates(query, vs)


def main():
    print_banner()

    chroma_path = Path("chroma_db")
    if not chroma_path.exists():
        print("⚠️  No resume index found.")
        print("📄 Please add resumes to the /resumes folder first.")
        choice = input("Build index now? (y/n): ").strip().lower()
        if choice == "y":
            build_vector_store()
        else:
            print("Exiting — please add resumes and restart.")
            sys.exit(0)

    while True:
        print_menu()
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            print("\n📄 Rebuilding resume index...")
            build_vector_store(force_rebuild=True)
            print("✅ Index rebuilt successfully!")

        elif choice == "2":
            job_desc = get_job_description()
            if not job_desc.strip():
                print("⚠️  No job description entered. Try again.")
                continue
            query = get_search_query()
            if not query.strip():
                print("⚠️  No search query entered. Try again.")
                continue
            run_pipeline(job_desc, query)

        elif choice == "3":
            query = get_search_query()
            if not query.strip():
                print("⚠️  No search query entered. Try again.")
                continue
            search_only(query)

        elif choice == "4":
            print("\n👋 Goodbye!\n")
            sys.exit(0)

        else:
            print("⚠️  Invalid choice. Please enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()
