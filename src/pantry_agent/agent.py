"""
Pantry Agent – Main LangGraph StateGraph Assembly
==================================================

Graph topology:

    START
      │
      ▼
    load_memory          ← Loads long-term user preferences & history
      │
      ▼
    classify_intent      ← LLM classifies intent + domain; detects images
      │
      ▼
    agent  ◄─────────────────────────┐
      │                              │
      │ tool_calls?                  │
      ├─── YES ──► tools ────────────┘
      │
      └─── NO ──► validate_output
                      │
                      │ approval_required?
                      ├─── YES ──► request_approval ──►┐
                      │                                │
                      └─── NO ──────────────────────►──┘
                                                        │
                                                   update_memory
                                                        │
                                                       END

Human-in-the-Loop:
  The graph is compiled with `interrupt_before=["request_approval"]`.
  When the node is reached, execution PAUSES. The caller resumes via:

      from langgraph.types import Command
      app.invoke(Command(resume={"approved": True}), config=config)

Usage:

    from pantry_agent.agent import create_agent

    app = create_agent()
    config = {"configurable": {"thread_id": "user-123", "user_id": "user-123"}}

    result = app.invoke(
        {
            "messages": [("human", "Scan my fridge and suggest recipes")],
            "user_input": "Scan my fridge and suggest recipes",
            "uploaded_images": [],
        },
        config=config,
    )
    print(result["messages"][-1].content)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import settings
from .nodes.agent_node import create_agent_node, should_continue
from .nodes.human_approval import request_approval, route_after_approval
from .nodes.intent_classifier import classify_intent
from .nodes.memory_loader import load_memory
from .nodes.memory_updater import update_memory
from .nodes.response_composer import compose_response
from .nodes.validator import needs_approval, validate_output
from .observability.tracing import configure_logging, log_agent_run
from .state import PantryAgentState, default_state
from .tools.registry import get_all_tools

logger = logging.getLogger(__name__)


def create_agent(
    checkpointer: Any = None,
    *,
    log_level: str = "INFO",
) -> Any:
    """
    Build and compile the Pantry LangGraph agent.

    Parameters
    ----------
    checkpointer
        A LangGraph checkpointer for persistence. Defaults to MemorySaver
        (in-process, non-persistent). Pass a SqliteSaver or RedisSaver
        for production persistence.
    log_level
        Logging verbosity for the pantry_agent namespace.

    Returns
    -------
    A compiled LangGraph application ready for `.invoke()` / `.stream()`.
    """
    configure_logging(log_level)

    # ── Tools ─────────────────────────────────────────────────────────────
    all_tools = get_all_tools()
    tool_node = ToolNode(tools=all_tools)

    # ── Agent node (ReAct core) ───────────────────────────────────────────
    # Pass the full tool list so the LLM's .bind_tools() always has every
    # tool available; domain-based filtering happens inside create_agent_node.
    agent_fn = create_agent_node(tools=all_tools)

    # ── Graph builder ─────────────────────────────────────────────────────
    builder: StateGraph = StateGraph(PantryAgentState)

    # Nodes
    builder.add_node("load_memory", load_memory)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("agent", agent_fn)
    builder.add_node("tools", tool_node)
    builder.add_node("validate_output", validate_output)
    builder.add_node("request_approval", request_approval)
    builder.add_node("update_memory", update_memory)
    builder.add_node("compose_response", compose_response)

    # ── Edges ─────────────────────────────────────────────────────────────
    # Linear entry path
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "classify_intent")
    builder.add_edge("classify_intent", "agent")

    # ReAct loop: agent ↔ tools
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "validate_output": "validate_output",
        },
    )
    builder.add_edge("tools", "agent")

    # Post-execution path
    builder.add_conditional_edges(
        "validate_output",
        needs_approval,
        {
            "request_approval": "request_approval",
            "update_memory": "update_memory",
        },
    )
    builder.add_edge("request_approval", "update_memory")
    builder.add_edge("update_memory", "compose_response")
    builder.add_edge("compose_response", END)

    # ── Compile ───────────────────────────────────────────────────────────
    if checkpointer is None:
        checkpointer = MemorySaver()

    app = builder.compile(
        checkpointer=checkpointer,
        # Graph pauses BEFORE this node – human gets to review before any
        # approval logic inside the node runs.
        interrupt_before=["request_approval"],
    )

    logger.info(
        "Pantry agent compiled: %d tools, model=%s",
        len(all_tools),
        settings.openai_model,
    )
    return app


def run_agent(
    user_message: str,
    thread_id: str = "default",
    user_id: str = "default_user",
    uploaded_images: list[str] | None = None,
    app: Any = None,
) -> dict[str, Any]:
    """
    Convenience wrapper: create (or reuse) an agent and invoke it with a
    single user message.

    Returns the final state dict.
    """
    if app is None:
        app = create_agent()

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    initial_state = default_state()
    initial_state["messages"] = [("human", user_message)]
    initial_state["user_input"] = user_message
    if uploaded_images:
        initial_state["uploaded_images"] = uploaded_images

    start = time.perf_counter()
    result = app.invoke(initial_state, config=config)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    log_agent_run(thread_id, user_message, result, elapsed_ms)
    return result
