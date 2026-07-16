import os
from datetime import date
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph

from skills_loader import load_skill
from tools import (
    check_github_profile,
    display_report_on_screen,
    evaluate_candidates_batch,
    search_resumes,
    send_shortlist_email,
)

load_dotenv()

project = os.getenv("GOOGLE_CLOUD_PROJECT")
region = os.getenv("GOOGLE_CLOUD_REGION")


# ─────────────────────────────────────────────
# 1. SHARED STATE — passed between all agents
# ─────────────────────────────────────────────
class RecruitState(TypedDict):
    job_description: str
    query: str
    retrieved_candidates: List[dict]
    evaluated_candidates: List[dict]
    final_report: str
    delivery_status: Optional[str]


# ─────────────────────────────────────────────
# 2. LLM — shared across all agents
# ─────────────────────────────────────────────
llm = ChatVertexAI(
    model="gemini-2.5-flash",
    project=project,
    location=region,
    temperature=0.2,
)
# For evaluation — fully deterministic
llm_eval = ChatVertexAI(
    model="gemini-2.5-flash",
    project=project,
    location=region,
    temperature=0,  # zero randomness for scoring
)

# For report writing — slight creativity allowed
llm_report = ChatVertexAI(
    model="gemini-2.5-flash",
    project=project,
    location=region,
    temperature=0.2,
)


# ─────────────────────────────────────────────
# 3. AGENT 1 — Retriever
# Searches vector store for relevant candidates
# Skill: skills/candidate_search/SKILL.md
# ─────────────────────────────────────────────
def retriever_agent(state: RecruitState) -> RecruitState:
    print("\n🤖 Agent 1 — Retriever running...")

    llm_retriever = llm.bind_tools([search_resumes])  # only the tool this agent needs

    response = llm_retriever.invoke(
        [
            SystemMessage(content=load_skill("candidate_search")),
            HumanMessage(
                content=(
                    f"Find candidates matching: {state['query']}. "
                    f"You must call the search_resumes tool with k=6 to do this — "
                    f"do not answer without calling it."
                )
            ),
        ]
    )

    if not response.tool_calls:
        print(
            "⚠️ Retriever LLM did not call search_resumes — falling back to direct search"
        )
        candidates = search_resumes.invoke({"query": state["query"], "k": 6})
    else:
        candidates = search_resumes.invoke(response.tool_calls[0]["args"])

    print(f"✅ Retriever found {len(candidates)} candidate(s)")
    return {**state, "retrieved_candidates": candidates}


# ─────────────────────────────────────────────
# 4. AGENT 2 — Evaluator
# Scores each candidate against job description
# Skill: skills/candidate_scoring/SKILL.md
# (loaded inside evaluate_candidates_batch, in tools.py —
#  this agent's body is unchanged, it only orchestrates)
# ─────────────────────────────────────────────
def evaluator_agent(state: RecruitState) -> RecruitState:
    print("\n🤖 Agent 2 — Evaluator running...")

    candidates = state["retrieved_candidates"]

    # Deterministic GitHub enrichment — no LLM decision involved
    enriched = []
    for c in candidates:
        github_username = c.get("github_username")  # set by ingestor, Phase A.5
        github_data = None
        if github_username:
            result = check_github_profile.invoke({"github_username": github_username})
            if "error" not in result:
                github_data = result
        enriched.append({**c, "github_data": github_data})

    evaluated = evaluate_candidates_batch(enriched, state["job_description"], llm_eval)
    evaluated.sort(key=lambda x: x["score"], reverse=True)

    return {**state, "evaluated_candidates": evaluated}


# ─────────────────────────────────────────────
# 5. AGENT 3 — Summarizer
# Produces final shortlist report
# Skill: skills/shortlist_reporting/SKILL.md
# ─────────────────────────────────────────────
def summarizer_agent(state: RecruitState) -> RecruitState:
    print("\n🤖 Agent 3 — Summarizer running...")

    if not state["evaluated_candidates"]:
        print("⚠️ No evaluated candidates — skipping report")
        return {
            **state,
            "final_report": "No candidates found matching the criteria. The database may be empty.",
        }

    candidates_text = ""
    for i, c in enumerate(state["evaluated_candidates"]):
        candidates_text += f"""
CANDIDATE {i + 1}: {c["source"]}
Score: {c["score"]}/10
{c["evaluation"]}
{"=" * 50}
"""

    prompt = f"""
Today's date is {date.today().strftime("%B %d, %Y")}.

JOB DESCRIPTION:
{state["job_description"]}

EVALUATED CANDIDATES:
{candidates_text}
"""

    response = llm_report.invoke(
        [
            SystemMessage(content=load_skill("shortlist_reporting")),
            HumanMessage(content=prompt),
        ]
    )

    print("✅ Summarizer complete")
    return {**state, "final_report": response.content}  # type: ignore


# ─────────────────────────────────────────────
# 6. AGENT 4 — Delivery_Agent
# Deliver the final report
# Skill: skills/result_delivery/SKILL.md
# ─────────────────────────────────────────────
import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def delivery_agent(state: RecruitState) -> RecruitState:
    print("\n🤖 Agent 4 — Delivery running...")

    user_answer = input(
        "\nHow would you like the results? "
        "(e.g. 'show it on screen' or 'email it to jane@company.com'): "
    )

    llm_delivery = llm.bind_tools([send_shortlist_email, display_report_on_screen])
    response = llm_delivery.invoke(
        [
            SystemMessage(content=load_skill("result_delivery")),
            HumanMessage(
                content=(
                    f'The user said: "{user_answer}". Based on this, call exactly one tool: '
                    f"send_shortlist_email if they gave an email address, otherwise "
                    f"display_report_on_screen. The report to deliver is provided separately, "
                    f"pass it as the shortlist_report argument. shortlist_report: {state['final_report']}"
                )
            ),
        ]
    )

    if not response.tool_calls:
        print("⚠️ Could not determine delivery preference — defaulting to screen")
        delivery_status = display_report_on_screen.invoke(
            {"shortlist_report": state["final_report"]}
        )
    else:
        call = response.tool_calls[0]
        args = call["args"]

        if call["name"] == "send_shortlist_email":
            # Real send — re-validate the address against the raw input rather
            # than trusting the LLM's extracted argument blindly.
            match = EMAIL_RE.search(user_answer)
            if not match:
                print(
                    "⚠️ Email requested but no valid address found in input — defaulting to screen"
                )
                delivery_status = display_report_on_screen.invoke(
                    {"shortlist_report": state["final_report"]}
                )
            else:
                print("\n💾 Sending a report to email .....")
                args["recipient"] = match.group(0)
                delivery_status = send_shortlist_email.invoke(args)
        else:
            delivery_status = display_report_on_screen.invoke(args)

    print(f"📬 {delivery_status}")
    return {**state, "delivery_status": delivery_status}


# ─────────────────────────────────────────────
# 7. BUILD THE GRAPH
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(RecruitState)

    # Add agent nodes
    graph.add_node("retriever", retriever_agent)
    graph.add_node("evaluator", evaluator_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("deliver", delivery_agent)

    # Wire the flow
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "evaluator")
    graph.add_edge("evaluator", "summarizer")
    graph.add_edge("summarizer", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 7. RUN THE PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build the graph
    app = build_graph()

    # Define the job description
    job_description = """
    We are seeking a Senior AI Architect with:
    - 10+ years of software engineering experience
    - Strong Python programming skills
    - Google Cloud Platform (GCP) expertise
    - Experience with AI/ML, LLMs, and RAG systems
    - Knowledge of Kubernetes, Docker, and microservices
    - Experience with LangChain or similar AI frameworks
    """

    # Initial state
    initial_state = RecruitState(
        job_description=job_description,
        query="GCP AI architect Python Kubernetes experience",
        retrieved_candidates=[],
        evaluated_candidates=[],
        final_report="",
        delivery_status="",
    )

    print("🚀 Starting Multi-Agent Recruitment Pipeline...")
    print("=" * 60)

    # Run the graph
    result = app.invoke(initial_state)

    # Print final report
    print("\n" + "=" * 60)
    print("📋 FINAL RECRUITMENT REPORT")
    print("=" * 60)
    print(result["final_report"])
