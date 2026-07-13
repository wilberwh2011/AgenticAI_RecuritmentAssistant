import json
import os
import smtplib
from email.mime.text import MIMEText
from typing import Dict, List

import requests
from langchain_core.tools import tool


@tool
def search_resumes(query: str, k: int = 6) -> List[Dict]:
    """Search the resume vector store for candidates matching a query.
    Returns a list of {source, content} matching the original retriever's schema.
    """
    from rag_engine import load_vector_store, search_candidates

    vectorstore = load_vector_store()
    results = search_candidates(query, vectorstore, k=k)
    return [
        {"source": doc.metadata.get("source", "unknown"), "content": doc.page_content}
        for doc in results
    ]


@tool
def check_github_profile(github_username: str) -> Dict:
    """Look up a candidate's public GitHub activity: repos, languages, bio.
    Called deterministically by the evaluator when a candidate has a GitHub
    username on file — never left to the LLM to decide whether to call.
    """
    resp = requests.get(f"https://api.github.com/users/{github_username}")
    if resp.status_code != 200:
        return {"error": f"GitHub user not found: {github_username}"}
    data = resp.json()
    repos = requests.get(data["repos_url"]).json()
    return {
        "username": github_username,
        "public_repos": data.get("public_repos"),
        "top_languages": list({r["language"] for r in repos if r.get("language")})[:5],
        "bio": data.get("bio"),
    }


def evaluate_candidates_batch(
    candidates: List[Dict], job_description: str, llm_eval
) -> List[Dict]:
    """Score ALL candidates relative to each other in a single call, for
    scoring consistency. This is deterministic business logic, not an
    LLM-chosen tool — it always runs, once, for the whole candidate set.
    Kept as a plain function (not bind_tools) for that reason; still
    exposable via the MCP server in Phase C as a callable endpoint.
    """
    all_candidates_text = ""
    for i, c in enumerate(candidates):
        github_note = ""
        if c.get("github_data"):
            gd = c["github_data"]
            github_note = f"\nGitHub activity: {gd['public_repos']} public repos, languages: {gd.get('top_languages')}"
        all_candidates_text += f"""
CANDIDATE {i + 1}: {c["source"]}
{c["content"]}{github_note}
{"=" * 40}
"""

    prompt = f"""
You are a strict senior technical recruiter comparing multiple candidates.
Evaluate ALL candidates RELATIVE to each other against the job description.
Be consistent — if Candidate A is clearly stronger than B, A must score higher.

Scoring rubric (apply strictly):
- 9-10: Exceeds ALL requirements, extremely rare
- 7-8: Meets most requirements, strong candidate
- 5-6: Meets some requirements, average candidate
- 3-4: Meets few requirements, weak candidate
- 1-2: Does not meet requirements

JOB DESCRIPTION:
{job_description}

CANDIDATES TO EVALUATE:
{all_candidates_text}

Respond ONLY as a JSON array, one object per candidate, in the same order given, no other text:
[{{"score": <1-10>, "meets_requirements": <true|false>, "strengths": ["...", "..."], "gaps": ["..."], "summary": "..."}}]
"""

    response = llm_eval.invoke(prompt)
    try:
        results = json.loads(response.content.strip().strip("```json").strip("```"))
    except Exception as e:
        print(f"⚠️ Failed to parse evaluator JSON: {e}")
        results = [
            {
                "score": 0,
                "meets_requirements": False,
                "strengths": [],
                "gaps": [],
                "summary": "Parse error",
            }
        ] * len(candidates)

    evaluated = []
    for c, r in zip(candidates, results):
        evaluated.append(
            {
                "source": c["source"],
                "content": c["content"],
                "evaluation": r,
                "score": r.get("score", 0),
            }
        )
        print(f"  ✅ Evaluated {c['source']} — Score: {r.get('score', 0)}/10")
    return evaluated


@tool
def send_shortlist_email(shortlist_report: str, recipient: str) -> str:
    """Send the final shortlist report by email to the given recipient.
    Real send — the recipient argument is honored as-is. Callers must
    validate the address before invoking this (see delivery_agent's regex
    safety net) since this function sends without further confirmation.
    """
    msg = MIMEText(shortlist_report)
    msg["Subject"] = "Candidate Shortlist Report"
    msg["From"] = os.environ["DEMO_EMAIL_SENDER"]
    msg["To"] = recipient
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ["DEMO_EMAIL_SENDER"], os.environ["DEMO_EMAIL_APP_PASSWORD"])
        server.send_message(msg)
    return f"Email sent to {recipient}"

@tool
def display_report_on_screen(shortlist_report: str) -> str:
    """Display the shortlist report directly in the terminal/UI instead of emailing it."""
    print("\n" + "=" * 60)
    print(shortlist_report)
    print("=" * 60)
    return "Report displayed on screen"
