"""
Memory Updater Node
===================
Last node before END. Extracts learnable facts from the current turn
and merges them into the user's long-term memory store.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from ..config import settings
from ..memory.long_term import load_user_memory, merge_memory_updates, save_user_memory
from ..prompts.system_prompts import MEMORY_SUMMARY_PROMPT
from ..state import PantryAgentState

logger = logging.getLogger(__name__)

_DEFAULT_USER_ID = "default_user"

_summariser_llm = ChatOpenAI(
    **settings.chat_openai_kwargs(
        temperature=0.0,
        max_tokens=512,
    )
)


def update_memory(
    state: PantryAgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """
    Extract facts from the current conversation turn and persist them
    to long-term memory.

    Only runs the LLM summariser when there are ≥2 messages, to avoid
    wasting tokens on trivial one-liners.
    """
    user_id: str = _DEFAULT_USER_ID
    if config and (cfg := config.get("configurable")):
        user_id = cfg.get("user_id", _DEFAULT_USER_ID)

    messages = state.get("messages", [])

    # ── Only summarise substantive conversations ──────────────────────────
    if len(messages) < 2:
        logger.debug("Skipping memory update – conversation too short")
        return {
            "execution_trace": state.get("execution_trace", [])
            + [{"node": "update_memory", "action": "skipped"}]
        }

    # ── Build conversation text for summariser ────────────────────────────
    conversation_lines: list[str] = []
    for msg in messages[-20:]:  # last 20 messages to stay within token limits
        if isinstance(msg, HumanMessage):
            conversation_lines.append(f"User: {msg.content}")
        elif hasattr(msg, "content") and msg.content:
            if isinstance(msg.content, str):
                conversation_lines.append(f"Assistant: {msg.content}")

    conversation = "\n".join(conversation_lines)

    # ── LLM summarisation ─────────────────────────────────────────────────
    try:
        prompt = MEMORY_SUMMARY_PROMPT.format(conversation=conversation)
        response = _summariser_llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content="Summarise now.")]
        )
        raw = response.content if isinstance(response.content, str) else "{}"
        # Strip markdown fences
        raw = raw.strip().strip("`").replace("json\n", "").replace("```", "")
        delta: dict[str, Any] = json.loads(raw)
    except Exception as exc:
        logger.warning("Memory summarisation failed: %s – skipping update", exc)
        return {
            "execution_trace": state.get("execution_trace", [])
            + [{"node": "update_memory", "action": "failed", "error": str(exc)}]
        }

    # ── Merge into long-term store ────────────────────────────────────────
    existing = load_user_memory(user_id)
    updated = merge_memory_updates(existing, delta)

    # Also capture the current pantry snapshot if available
    if state.get("pantry_items"):
        updated["pantry_snapshot"] = state["pantry_items"]

    save_user_memory(user_id, updated)
    logger.info(
        "Long-term memory updated for %s: %d preferences, %d recipes",
        user_id,
        len(updated.get("dietary_preferences", [])),
        len(updated.get("favourite_recipes", [])),
    )

    return {
        "memory": updated,
        "execution_trace": state.get("execution_trace", [])
        + [
            {
                "node": "update_memory",
                "action": "saved",
                "preferences": updated.get("dietary_preferences"),
            }
        ],
    }
