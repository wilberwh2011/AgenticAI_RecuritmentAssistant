"""Structured JSON logging for Cloud Logging, correlated with the OTel trace.

Every log line carries `trace_id`, `agent_node`, and `session_id` so a single
request can be filtered end-to-end in Cloud Logging, and the `trace_id` lines
up with whatever Cloud Trace / Langfuse span was active when the log was
emitted (see otel_setup.py — both exporters hang off the same provider, so
one trace_id resolves in both places).

Usage
-----
    from log_setup import init_logging, get_logger, bind_session_id, bind_agent_node

    init_logging()                      # once, at process startup
    log = get_logger(__name__)

    bind_session_id("api-abc123")       # once per incoming request
    bind_agent_node("retriever")        # once per graph node
    log.info("node_enter", query=state["query"])
"""

import contextvars
import logging
import sys

import structlog
from opentelemetry import trace

# ─────────────────────────────────────────────
# Per-request / per-node context
# Contextvars so agent_node/session_id ride along on every log call in this
# task without threading them through every function signature. Each async
# task (FastAPI request, LangGraph node) gets its own copy via asyncio's
# contextvar propagation, so concurrent requests don't bleed into each other.
# ─────────────────────────────────────────────
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="-"
)
_agent_node_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_node", default="-"
)


def bind_session_id(session_id: str) -> None:
    """Call once per incoming request (API) or per CLI run."""
    _session_id_var.set(session_id)


def bind_agent_node(agent_node: str) -> None:
    """Call on entry to each LangGraph node so every log line emitted while
    that node is running is tagged with it automatically."""
    _agent_node_var.set(agent_node)


def _add_run_context(logger, method_name, event_dict):
    event_dict.setdefault("session_id", _session_id_var.get())
    event_dict.setdefault("agent_node", _agent_node_var.get())
    return event_dict


def _add_trace_context(logger, method_name, event_dict):
    """Pull trace_id/span_id off whatever OTel span is currently active.
    otel_setup.py attaches both the Cloud Trace and Langfuse exporters to
    the same global provider, so this trace_id is the join key across
    Cloud Logging, Cloud Trace, and Langfuse for a single request.
    """
    span_ctx = trace.get_current_span().get_span_context()
    if span_ctx.is_valid:
        event_dict["trace_id"] = format(span_ctx.trace_id, "032x")
        event_dict["span_id"] = format(span_ctx.span_id, "016x")
    else:
        event_dict["trace_id"] = "-"
        event_dict["span_id"] = "-"
    return event_dict


_configured = False


def init_logging(level: int = logging.INFO) -> None:
    """Call once at process startup. Idempotent — safe to call from both
    main() (CLI path) and the FastAPI startup event without double-configuring.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_run_context,
            _add_trace_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Cloud Logging's JSON parser looks for a "message" field — give
            # it what it wants instead of "event", structlog's default.
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "recruitment-assistant"):
    return structlog.get_logger(name)


# ─────────────────────────────────────────────
# State-diff helper for agent-boundary logs
# ─────────────────────────────────────────────
def diff_state(before: dict, after: dict) -> dict:
    """Shallow diff between a LangGraph node's input/output state, for
    node_exit logs. Only reports keys that actually changed, and summarizes
    big values (candidate lists, report text) instead of dumping full
    content — keeps each log line a reasonable size for Cloud Logging
    rather than a multi-KB blob per node.
    """
    changed = {}
    for key in after:
        before_val = before.get(key)
        after_val = after.get(key)
        if before_val == after_val:
            continue
        changed[key] = {
            "before": _summarize(before_val),
            "after": _summarize(after_val),
        }
    return changed


def _summarize(value):
    if isinstance(value, list):
        return f"<list len={len(value)}>"
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + f"...<truncated, {len(value)} chars total>"
    return value
