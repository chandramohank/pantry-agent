"""
Observability & Tracing
=======================
Configures LangSmith tracing and provides structured logging utilities
for monitoring agent behaviour in production.

Tracing is opt-in: set LANGCHAIN_TRACING_V2=true in .env to enable.

Structured execution traces are always written to the agent state's
`execution_trace` list, regardless of LangSmith configuration.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


# ── LangSmith setup ───────────────────────────────────────────────────────────
# LangSmith picks up configuration via environment variables set in config.py's
# @model_validator. No explicit SDK calls are required here.


def log_agent_run(
    thread_id: str,
    user_input: str,
    final_state: dict[str, Any],
    elapsed_ms: int,
) -> None:
    """
    Emit a structured JSON-compatible log entry for each agent invocation.
    Consumed by log aggregators (CloudWatch, Datadog, Elastic, etc.).
    """
    trace = final_state.get("execution_trace", [])
    tool_names = [
        step.get("tool_calls", [])
        for step in trace
        if step.get("node") == "agent"
    ]
    # Flatten list-of-lists
    tools_used = [t for sublist in tool_names for t in sublist]

    logger.info(
        "agent_run",
        extra={
            "thread_id": thread_id,
            "intent": final_state.get("intent"),
            "domain": final_state.get("domain"),
            "tools_used": tools_used,
            "validation_errors": final_state.get("validation_errors", []),
            "human_approval_required": final_state.get("human_approval_required"),
            "elapsed_ms": elapsed_ms,
            "node_count": len(trace),
        },
    )


@contextmanager
def timed_operation(label: str) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = int((time.perf_counter() - start) * 1000)
        logger.debug("Timed [%s]: %d ms", label, elapsed)


def traced_node(node_name: str) -> Callable:
    """
    Decorator for graph nodes that adds automatic timing and error logging.

    Usage:
        @traced_node("my_node")
        def my_node(state):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = int((time.perf_counter() - start) * 1000)
                logger.debug("Node [%s] completed in %d ms", node_name, elapsed)
                return result
            except Exception as exc:
                elapsed = int((time.perf_counter() - start) * 1000)
                logger.error(
                    "Node [%s] failed after %d ms: %s",
                    node_name,
                    elapsed,
                    exc,
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def configure_logging(level: str = "INFO") -> None:
    """Set up structured logging for the pantry agent."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
