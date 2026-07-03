"""
Memory Loader Node
==================
First node in the graph. Loads long-term user memory so all subsequent
nodes can reference dietary preferences, past interactions, etc.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..memory.long_term import load_user_memory
from ..state import PantryAgentState

logger = logging.getLogger(__name__)

_DEFAULT_USER_ID = "default_user"


def load_memory(
    state: PantryAgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """
    Load long-term memory for the current user into agent state.

    The user_id is read from the LangGraph config's 'configurable' dict.
    Falls back to 'default_user' when not provided.
    """
    user_id: str = _DEFAULT_USER_ID
    if config and (cfg := config.get("configurable")):
        user_id = cfg.get("user_id", _DEFAULT_USER_ID)

    memory = load_user_memory(user_id)
    logger.debug("Memory loaded for user %s (session #%d)", user_id, memory.get("session_count", 0))

    return {
        "memory": memory,
        "execution_trace": state.get("execution_trace", [])
        + [{"node": "load_memory", "user_id": user_id}],
    }
