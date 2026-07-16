# Candidate Search Skill

## When to use
Whenever the workflow needs to find candidates from the resume vector store
matching a query — this is the retriever agent's only responsibility.

## Tools available
- search_resumes(query: str, k: int = 6) -> List[{source, content}]
  Bound via bind_tools — the LLM decides the exact query text and k value
  based on the user's original request, then the tool executes the vector
  search and returns matching resume chunks.

## How this differs from a deterministic step
This is one of the few steps in the graph where the LLM is given a real
choice (how to phrase the search query). Because that choice can fail —
the LLM may respond with plain text instead of calling the tool — this
skill REQUIRES a fallback: if response.tool_calls is empty, call
search_resumes directly with the raw user query and k=6. Never let the
retriever return zero candidates silently just because the LLM didn't
call the tool.

## Guidance
- Always instruct the LLM explicitly to call search_resumes — do not rely
  on it inferring this from context alone.
- Pin k=6 by default unless the user's request implies a broader or
  narrower search (e.g. "just show me a couple of options" -> lower k).
- Do not attempt to rank or filter candidates here — that is the
  evaluator's job. This skill only retrieves.

## Output
List of dicts: {source: str, content: str}. No scoring, no filtering.
