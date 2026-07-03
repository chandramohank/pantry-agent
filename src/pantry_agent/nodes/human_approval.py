"""
Human Approval Node
===================
Implements a human-in-the-loop gate using LangGraph's `interrupt()`.

When the graph reaches this node execution is PAUSED and the calling
application receives a payload describing what needs review.

To resume, the caller invokes:

    app.invoke(
        Command(resume={"approved": True, "feedback": "Looks good"}),
        config={"configurable": {"thread_id": "<thread>"}},
    )

Or to reject:

    app.invoke(
        Command(resume={"approved": False, "feedback": "Wrong quantities"}),
        ...
    )
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from ..prompts.system_prompts import HUMAN_APPROVAL_MESSAGE
from ..state import PantryAgentState

logger = logging.getLogger(__name__)


def request_approval(state: PantryAgentState) -> dict[str, Any]:
    """
    Pause execution and ask the human to approve or reject the pending action.

    The `interrupt()` call serialises the approval payload and suspends
    the graph until `Command(resume=...)` is received.
    """
    reason = state.get("approval_reason", "Action requires review.")
    items = state.get("extracted_items", [])

    # Build a human-readable items summary
    if items:
        items_formatted = "\n".join(
            f"  • {it.get('name', '?')} — {it.get('quantity', '?')} {it.get('unit', '')}"
            for it in items[:20]
        )
        if len(items) > 20:
            items_formatted += f"\n  … and {len(items) - 20} more"
    else:
        items_formatted = "  (no specific items — see conversation for context)"

    action = f"Pantry update triggered by intent: {state.get('intent', 'unknown')}"

    message = HUMAN_APPROVAL_MESSAGE.format(
        reason=reason,
        action=action,
        items_formatted=items_formatted,
    )

    approval_payload = {
        "reason": reason,
        "action": action,
        "items": items,
        "message": message,
    }

    logger.info("Human approval requested: %s", reason)

    # ── This suspends the graph ───────────────────────────────────────────
    result: dict[str, Any] = interrupt(approval_payload)

    # ── Execution resumes here after Command(resume=...) ─────────────────
    approved: bool = bool(result.get("approved", False))
    feedback: str = result.get("feedback", "")

    logger.info("Human decision: approved=%s, feedback=%s", approved, feedback or "(none)")

    # If rejected, clear items to prevent silent saves downstream
    updates: dict[str, Any] = {
        "human_approved": approved,
        "human_approval_required": False,
        "execution_trace": state.get("execution_trace", [])
        + [
            {
                "node": "request_approval",
                "approved": approved,
                "feedback": feedback,
            }
        ],
    }

    if not approved:
        updates["extracted_items"] = []
        updates["messages"] = [
            AIMessage(
                content=(
                    f"The action was not approved. Reason: {feedback or 'No reason provided.'} "
                    "No changes were saved to your pantry."
                )
            )
        ]

    return updates


def route_after_approval(state: PantryAgentState) -> str:
    """Conditional edge after human approval node."""
    return "update_memory"


def _format_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return json.dumps([])
    return json.dumps(
        [{"name": it.get("name"), "qty": it.get("quantity"), "unit": it.get("unit")} for it in items],
        indent=2,
    )
