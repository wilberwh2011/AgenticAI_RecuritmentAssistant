# Result Delivery Skill

## When to use
Once a final_report exists and needs to be delivered to the user — this
is the delivery agent's only responsibility, and runs last in the graph.

## Tools available
- send_shortlist_email(shortlist_report: str, recipient: str) -> str
  REAL SEND — no demo-mode redirect. Sends to whatever recipient
  argument it's given, with no further confirmation step inside the tool
  itself.
- display_report_on_screen(shortlist_report: str) -> str
  Prints the report to the terminal/UI. No side effects, always safe.
Both are bound via bind_tools — this is the one skill in the whole
workflow where the LLM is deliberately given a real choice between two
tools, because the user's delivery preference is genuinely free text.

## How this differs from other skills
Unlike candidate_scoring or shortlist_reporting, this skill's core job IS
the LLM decision (screen vs. email, and if email, which address). That
decision is not deterministic and should not be hardcoded.

## Safety requirement — do not skip
Because send_shortlist_email is a real send, NEVER trust the LLM's
extracted recipient argument directly. Always re-validate the email
address against the raw user input using a regex
([\w.+-]+@[\w-]+\.[\w.-]+) before calling send_shortlist_email. If no
valid address is found in the raw input despite the LLM choosing
send_shortlist_email, fall back to display_report_on_screen rather than
sending to a possibly-hallucinated address.

## Guidance
- If response.tool_calls is empty (LLM didn't choose either tool),
  default to display_report_on_screen — never leave the user without any
  output.
- Do not add a demo-mode/redirect-to-self shortcut back into this skill;
  that was deliberately removed in favor of real sends. If a safer mode
  is needed again later, it should be an explicit environment flag, not
  silent tool behavior.

## Output
A single string describing the delivery outcome (delivery_status), e.g.
"Email has been sent to jane@company.com" or "Report displayed on screen".
