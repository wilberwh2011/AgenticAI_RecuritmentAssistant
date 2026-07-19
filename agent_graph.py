import asyncio
import json
import os
from datetime import date
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph

from mcp_client_config import mcp_client
from skills_loader import load_skill
from tools import (
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


async def build_agent_tools():
    return await mcp_client.get_tools()


def make_evaluator_agent(github_search_repos_tool):
    """Factory — closes over the GitHub MCP tool fetched once at startup."""

    async def evaluator_agent(state: RecruitState) -> RecruitState:
        print("\n🤖 Agent 2 — Evaluator running...")

        candidates = state["retrieved_candidates"]

        enriched = []
        for c in candidates:
            github_username = c.get("github_username")
            github_data = None
            if github_username and github_search_repos_tool:
                result = await github_search_repos_tool.ainvoke(
                    {"query": f"user:{github_username}"}
                )

                data = None
                if isinstance(result, list) and result:
                    first_block = result[0]
                    if isinstance(first_block, dict) and "text" in first_block:
                        try:
                            data = json.loads(first_block["text"])
                        except json.JSONDecodeError:
                            data = None

                repos = data.get("items", []) if isinstance(data, dict) else []
                if repos:
                    github_data = {
                        "public_repos": data.get("total_count", len(repos)),
                        "repos": [
                            {"name": r["name"], "description": r.get("description")}
                            for r in repos
                        ],
                    }
            enriched.append({**c, "github_data": github_data})

        evaluated = evaluate_candidates_batch(
            enriched, state["job_description"], llm_eval
        )
        evaluated.sort(key=lambda x: x["score"], reverse=True)

        return {**state, "evaluated_candidates": evaluated}

    return evaluator_agent


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

    decision = decide_delivery(user_answer, state["final_report"])

    if decision["fallback_reason"]:
        print(f"⚠️ Falling back to screen delivery ({decision['fallback_reason']})")

    if decision["tool"] == "send_shortlist_email":
        print("\n💾 Sending a report to email .....")
        delivery_status = send_shortlist_email.invoke(decision["args"])
    else:
        delivery_status = display_report_on_screen.invoke(decision["args"])

    print(f"📬 {delivery_status}")
    return {**state, "delivery_status": delivery_status}


def decide_delivery(user_answer: str, final_report: str) -> dict:
    """Pure decision logic: given free-text delivery preference + the report,
    decide which tool to call and with what args. No I/O, no side effects —
    this is what the eval harness calls directly with fixture strings.

    Returns: {"tool": "send_shortlist_email" | "display_report_on_screen",
              "args": {...}, "fallback_reason": Optional[str]}
    fallback_reason is set whenever we defaulted to screen instead of honoring
    an apparent email request — useful for eval assertions and for logging.
    """
    llm_delivery = llm.bind_tools([send_shortlist_email, display_report_on_screen])
    response = llm_delivery.invoke(
        [
            SystemMessage(content=load_skill("result_delivery")),
            HumanMessage(
                content=(
                    f'The user said: "{user_answer}". Based on this, call exactly one tool: '
                    f"send_shortlist_email if they gave an email address, otherwise "
                    f"display_report_on_screen. The report to deliver is provided separately, "
                    f"pass it as the shortlist_report argument. shortlist_report: {final_report}"
                )
            ),
        ]
    )

    if not response.tool_calls:
        return {
            "tool": "display_report_on_screen",
            "args": {"shortlist_report": final_report},
            "fallback_reason": "no_tool_call",
        }

    call = response.tool_calls[0]
    args = call["args"]

    if call["name"] == "send_shortlist_email":
        # Real send — re-validate the address against the raw input rather
        # than trusting the LLM's extracted argument blindly.
        match = EMAIL_RE.search(user_answer)
        if not match:
            return {
                "tool": "display_report_on_screen",
                "args": {"shortlist_report": final_report},
                "fallback_reason": "email_requested_but_unparseable",
            }
        args["recipient"] = match.group(0)
        return {"tool": "send_shortlist_email", "args": args, "fallback_reason": None}

    return {"tool": "display_report_on_screen", "args": args, "fallback_reason": None}


# ─────────────────────────────────────────────
# 7. BUILD THE GRAPH
# ─────────────────────────────────────────────
async def build_graph():
    """Builds and returns a compiled graph. Callable from main.py or anywhere else."""
    mcp_tools = await build_agent_tools()
    github_search_repos_tool = next(
        (t for t in mcp_tools if t.name == "search_repositories"), None
    )
    if github_search_repos_tool is None:
        print(
            "⚠️ GitHub search_repositories tool not found — evaluator will skip GitHub enrichment"
        )

    graph = StateGraph(RecruitState)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("evaluator", make_evaluator_agent(github_search_repos_tool))
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("deliver", delivery_agent)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "evaluator")
    graph.add_edge("evaluator", "summarizer")
    graph.add_edge("summarizer", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()  # ← the fix: actually return it


# ─────────────────────────────────────────────
# 8. STANDALONE TEST — only runs when this file
# is executed directly (python agent_graph.py),
# NOT when main.py imports build_graph
# ─────────────────────────────────────────────
async def _standalone_test():
    app = await build_graph()

    job_description = """
    We are seeking a Senior AI Architect with:
    - 10+ years of software engineering experience
    - Strong Python programming skills
    - Google Cloud Platform (GCP) expertise
    - Experience with AI/ML, LLMs, and RAG systems
    - Knowledge of Kubernetes, Docker, and microservices
    - Experience with LangChain or similar AI frameworks
    """

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

    result = await app.ainvoke(initial_state)

    print("\n" + "=" * 60)
    print("📋 FINAL RECRUITMENT REPORT")
    print("=" * 60)
    print(result["final_report"])


if __name__ == "__main__":
    asyncio.run(_standalone_test())
