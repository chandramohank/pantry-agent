"""
Agent Node
==========
The ReAct core of the pantry assistant.

This module provides a factory function `create_agent_node` that returns a
LangGraph node function bound to a specific set of tools and a system prompt
that adapts to the classified intent / domain from earlier nodes.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..memory.short_term import build_memory_summary_context
from ..prompts.system_prompts import AGENT_SYSTEM_PROMPT
from ..state import PantryAgentState
from ..tools.registry import VISION_PRIORITY_TOOLS, get_tools_for_domain

logger = logging.getLogger(__name__)


def create_agent_node(tools: list | None = None) -> Callable[[PantryAgentState], dict[str, Any]]:
    """
    Factory that returns a LangGraph node function.

    If *tools* is None, the tool set is resolved dynamically at call-time
    based on state.domain. If provided, those tools are always used
    (useful for tests or specialised sub-graphs).
    """
    base_llm = ChatOpenAI(**settings.chat_openai_kwargs())

    def agent_node(state: PantryAgentState) -> dict[str, Any]:
        domain = state.get("domain", "General")
        intent = state.get("intent", "general_query")
        memory = state.get("memory", {})
        uploaded_images = state.get("uploaded_images", [])
        messages = list(state.get("messages", []))

        # ── Select tool set ───────────────────────────────────────────────
        if tools is not None:
            active_tools = tools
        elif uploaded_images:
            # Images always trigger vision-priority tool ordering
            active_tools = VISION_PRIORITY_TOOLS
        else:
            active_tools = get_tools_for_domain(domain)

        llm_with_tools = base_llm.bind_tools(active_tools)

        # ── Build dynamic system prompt ───────────────────────────────────
        memory_summary = build_memory_summary_context(memory)
        system_content = AGENT_SYSTEM_PROMPT.format(
            domain=domain,
            intent=intent,
            has_images=bool(uploaded_images),
            memory_summary=memory_summary,
            bulk_threshold=settings.human_approval_required_for_bulk,
        )

        all_messages = [SystemMessage(content=system_content)] + messages

        # ── LLM call ──────────────────────────────────────────────────────
        response: AIMessage = llm_with_tools.invoke(all_messages)

        # ── Execution trace ───────────────────────────────────────────────
        tool_call_names = [tc["name"] for tc in (response.tool_calls or [])]
        trace_entry: dict[str, Any] = {
            "node": "agent",
            "domain": domain,
            "intent": intent,
            "tool_calls": tool_call_names,
            "has_final_answer": not bool(response.tool_calls),
        }

        logger.info(
            "Agent: domain=%s, tool_calls=%s, final=%s",
            domain,
            tool_call_names or "none",
            not bool(response.tool_calls),
        )

        updates: dict[str, Any] = {
            "messages": [response],
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
        }

        # Surface selected_tool for observability
        if tool_call_names:
            updates["selected_tool"] = tool_call_names[0]

        return updates

    return agent_node


def should_continue(state: PantryAgentState) -> str:
    """
    Conditional edge: route to 'tools' if the agent issued tool calls,
    otherwise route to 'validate_output'.
    """
    messages = state.get("messages", [])
    if not messages:
        return "validate_output"

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "validate_output"
