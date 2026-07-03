"""
Output Validator Node
=====================
Examines the most recent tool output(s) in agent messages and decides:
1. Whether the data is valid and trustworthy
2. Whether human approval is required before proceeding

Confidence thresholds and bulk thresholds come from application settings.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from ..config import settings
from ..state import PantryAgentState

logger = logging.getLogger(__name__)

# ── Heuristic thresholds ──────────────────────────────────────────────────────
_CONFIDENCE_THRESHOLD = settings.vision_confidence_threshold
_BULK_THRESHOLD = settings.human_approval_required_for_bulk

# Tool names that touch persistent state and warrant stricter validation
_WRITE_TOOLS = {
    "add_pantry_item",
    "extract_and_save_pantry_items",
    "extract_ingredients_from_image",
}


def validate_output(state: PantryAgentState) -> dict[str, Any]:
    """
    Validate recent tool outputs and flag high-risk operations for human review.

    Sets: validation_errors, human_approval_required, approval_reason.
    """
    messages = state.get("messages", [])
    errors: list[str] = []
    approval_required = False
    approval_reason = ""

    # ── Collect recent tool messages ──────────────────────────────────────
    tool_outputs: list[dict[str, Any]] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else {}
            except json.JSONDecodeError:
                data = {"raw": msg.content}
            tool_outputs.append({"tool_name": msg.name, "data": data})
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            # Reached the final AI response – stop scanning backwards
            break

    for output in tool_outputs:
        tool_name: str = output.get("tool_name", "unknown")
        data: dict[str, Any] = output.get("data", {})

        # ── Error surfaced by the tool itself ─────────────────────────────
        if data.get("error"):
            errors.append(f"{tool_name}: {data.get('message', 'Unknown error')}")
            continue

        # ── Confidence check (vision and extraction tools) ────────────────
        confidence: float | None = data.get("confidence")
        if confidence is not None and confidence < _CONFIDENCE_THRESHOLD:
            approval_required = True
            approval_reason = (
                f"Vision/extraction confidence {confidence:.0%} is below "
                f"the {_CONFIDENCE_THRESHOLD:.0%} threshold. "
                "Please review the extracted items before they are saved."
            )
            logger.warning(
                "Low confidence extraction from %s: %.2f < %.2f",
                tool_name,
                confidence,
                _CONFIDENCE_THRESHOLD,
            )

        # ── Bulk import check ─────────────────────────────────────────────
        items: list = data.get("extracted_items", data.get("saved_items", []))
        if tool_name in _WRITE_TOOLS and len(items) >= _BULK_THRESHOLD:
            approval_required = True
            approval_reason = (
                f"Bulk operation: {len(items)} items are about to be saved. "
                "Please confirm before proceeding."
            )

        # ── Quantity sanity check ─────────────────────────────────────────
        for item in items:
            qty = item.get("quantity", 0)
            name = item.get("name", "unknown")
            if qty > 500:  # > 500 of anything is suspicious
                errors.append(
                    f"Suspicious quantity for '{name}': {qty}. Please verify."
                )

    # Carry forward existing pantry / recipe data if populated by tools
    updates: dict[str, Any] = {
        "validation_errors": errors,
        "human_approval_required": approval_required,
        "approval_reason": approval_reason,
        "execution_trace": state.get("execution_trace", [])
        + [
            {
                "node": "validate_output",
                "errors": errors,
                "approval_required": approval_required,
            }
        ],
    }

    # Extract and surface domain data from tool outputs
    for output in tool_outputs:
        data = output.get("data", {})
        tool_name = output.get("tool_name", "")

        if "extracted_items" in data:
            updates["extracted_items"] = data["extracted_items"]
        if "items" in data and tool_name == "get_pantry_inventory":
            updates["pantry_items"] = data["items"]
        if "recipes" in data:
            updates["recipes"] = data["recipes"]
        if "waste_items" in data:
            updates["waste_analysis"] = data["waste_items"]
        if "insights" in data:
            updates["sustainability_data"] = data

    if errors:
        logger.warning("Validation errors: %s", errors)

    return updates


def needs_approval(state: PantryAgentState) -> str:
    """Conditional edge: route to 'request_approval' or 'update_memory'."""
    return "request_approval" if state.get("human_approval_required") else "update_memory"
