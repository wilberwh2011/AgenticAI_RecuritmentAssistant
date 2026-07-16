# Candidate Scoring Skill

## When to use
Whenever a batch of retrieved candidates needs to be scored against a job
description — this is the evaluator agent's only responsibility.

## Tools available
- check_github_profile(github_username: str) -> {public_repos, top_languages, bio}
  Called directly via .invoke() — NOT offered to the LLM as a choosable
  tool. Called once per candidate, only when that candidate has a stored
  github_username. This is a deterministic enrichment step based on data
  presence, not an LLM decision.
- evaluate_candidates_batch(candidates, job_description, llm_eval) -> List[dict]
  Plain function, NOT offered via bind_tools. Always runs exactly once per
  workflow run, on the FULL candidate set together.

## How this differs from a normal tool
Do not refactor evaluate_candidates_batch into per-candidate LLM tool
calls. Relative scoring requires all candidates to be visible to the LLM
in the same call — if Candidate A is clearly stronger than B, A must
score higher. Splitting this into independent per-candidate calls breaks
that comparability guarantee, since each call would have no visibility
into the others' resumes.

## Deterministic sub-step: GitHub enrichment
For each candidate with a github_username on file, check_github_profile
is called BEFORE evaluate_candidates_batch runs, and the result is folded
into that candidate's prompt context as a "GitHub activity" line. If a
candidate has no github_username, this step is skipped for them — this
must never be treated as a negative signal; many strong candidates simply
don't have public repos.

## Guidance
- Apply the scoring rubric strictly (9-10 exceeds all requirements /
  7-8 meets most / 5-6 meets some / 3-4 meets few / 1-2 does not meet).
- GitHub activity, when present, should be treated as corroborating
  evidence for a claimed skill, not as an independent scoring factor.
- Response format must be a JSON array, one object per candidate, in the
  same order given — do not deviate from this schema, since the
  evaluator's parser depends on strict ordering and structure.

## Output
JSON array: [{score, meets_requirements, strengths, gaps, summary}, ...]
one object per candidate, same order as input.