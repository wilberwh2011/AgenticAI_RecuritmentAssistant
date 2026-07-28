## Logging

The app emits **structured JSON logs** (via `structlog`, see `log_setup.py`) so Cloud
Logging can parse fields natively instead of scraping free-text messages. Every log
line carries three correlation fields:

| Field        | Set by                                   | Purpose                                                        |
|--------------|-------------------------------------------|------------------------------------------------------------------|
| `trace_id`   | current OTel span (`otel_setup.py`)       | Ties a log line to the matching Cloud Trace span and Langfuse trace |
| `session_id` | `bind_session_id(...)` once per request   | Groups every log line for one `/run`, `/search`, or CLI invocation |
| `agent_node` | `bind_agent_node(...)` once per LangGraph node | Identifies which agent (`retriever`, `evaluator`, `summarizer`, `deliver`) emitted the line |

Each LangGraph node logs a `node_enter` event on entry and a `node_exit` event on
exit, with a `state_diff` showing only the state keys that changed (large values
like candidate lists or report text are summarized, not dumped in full, to keep
log lines a reasonable size).

### Viewing logs locally

Logs are written as JSON to stdout. Pipe through `jq` for readability:

```bash
python main.py | jq .
# or for the API:
uvicorn main:app --reload | jq .
```

### Viewing logs on Cloud Run

Cloud Run captures stdout/stderr automatically and ships it to Cloud Logging —
no extra exporter needed, which is why `log_setup.py` writes JSON straight to
stdout. Cloud Logging auto-detects the JSON and promotes each field (`session_id`,
`agent_node`, `trace_id`, etc.) into `jsonPayload`, so they're individually
queryable.

**Console:** Cloud Run → your service → **Logs** tab.

**Logs Explorer** (Cloud Console → Logging → Logs Explorer), or from the CLI:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="YOUR_SERVICE_NAME"' \
  --limit 50 --format json
```

**Tail live logs while testing:**

```bash
gcloud beta run services logs tail YOUR_SERVICE_NAME --region YOUR_REGION
```

### Useful Logs Explorer queries

Filter to one session (e.g. while debugging a specific `/run` call):

```
resource.type="cloud_run_revision"
jsonPayload.session_id="api-python gcp architect"
```

Follow a single request end-to-end across all four agent nodes, in order —
this is the payoff of the shared `trace_id`:

```
jsonPayload.trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
```

Find only agent-boundary transitions (skip the free-text `print()` noise):

```
resource.type="cloud_run_revision"
jsonPayload.message=("node_enter" OR "node_exit")
```

Because `session_id` and `trace_id` in these logs match what's sent to Langfuse
(same `session_id` string) and Cloud Trace (same `trace_id`), you can jump from
an odd-looking Langfuse trace straight to the matching structured logs (with
state diffs) by copying its id into the query above, or vice versa.

### Dependencies

`structlog` is required — add it to `requirements.txt` / `pyproject.toml` if not
already present:

```
pip install structlog
```
