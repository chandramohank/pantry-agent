"""
Short-term (in-session) memory
==============================
Wraps the LangGraph checkpointer's conversation history.
Provides helpers to extract a human-readable summary of the current thread.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def extract_conversation_turns(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert a LangGraph message list into simple {role, content} dicts."""
    turns: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            turns.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            turns.append({"role": "assistant", "content": str(msg.content)})
    return turns


def get_recent_turns(messages: list[BaseMessage], n: int = 10) -> list[dict[str, str]]:
    """Return the last *n* human/AI turn pairs."""
    turns = extract_conversation_turns(messages)
    return turns[-n:]


def build_memory_summary_context(memory: dict[str, Any]) -> str:
    """Format the long-term memory dict into a compact string for system prompts."""
    if not memory:
        return "No prior session data."

    parts: list[str] = []

    if prefs := memory.get("dietary_preferences"):
        parts.append(f"Diet preferences: {', '.join(prefs)}")

    if allergies := memory.get("allergies"):
        parts.append(f"Allergies / avoids: {', '.join(allergies)}")

    if recipes := memory.get("favourite_recipes"):
        sample = recipes[:3]
        parts.append(f"Favourite recipes: {', '.join(sample)}")

    if subs := memory.get("substitutions"):
        subs_str = "; ".join(f"{k}→{v}" for k, v in list(subs.items())[:3])
        parts.append(f"Known substitutions: {subs_str}")

    return " | ".join(parts) if parts else "No prior preferences recorded."
