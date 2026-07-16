# Shortlist Reporting Skill

## When to use
Whenever a set of evaluated (scored) candidates needs to be turned into a
final, human-readable report — this is the summarizer agent's only
responsibility.

## Tools available
None. This skill is entirely deterministic — a single LLM call that
generates prose from already-scored data. No tool binding, no decisions
about whether to call anything.

## How this differs from a normal tool
There is no "choice" involved here — if evaluated candidates exist, a
report is always generated. If evaluated_candidates is empty, this skill
short-circuits and returns a fixed fallback message instead of calling
the LLM at all ("No candidates found matching the criteria. The database
may be empty.") — do not remove this guard, it prevents a wasted/garbage
LLM call on empty input.

## Guidance
- Include today's date in the report header.
- Structure is fixed: (1) Executive Summary, (2) Recommended Candidates
  ranked with reasons, (3) Candidates to Decline with brief reason,
  (4) Suggested Interview Questions for the top candidate.
- This skill does NOT decide how the report is delivered (screen vs.
  email) — that decision belongs entirely to the result_delivery skill.
  Do not add delivery logic here.

## Output
A single string: the full formatted report (final_report).