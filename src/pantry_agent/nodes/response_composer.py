"""Response composer node.

Builds a tool-agnostic response envelope with renderable UI artifacts from the
validated state. This keeps tool invocation/reasoning decoupled from UI.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ..models.schemas import ActionKind, AgentResponseEnvelope, UIAction, UIArtifact, UILayout
from ..state import PantryAgentState


def _message_role(message: Any) -> str | None:
    msg_type = getattr(message, "type", None)
    if isinstance(msg_type, str):
        return msg_type
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role
    return None


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _last_assistant_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = _message_role(message)
        if role in {"ai", "assistant"}:
            return _message_content_text(getattr(message, "content", ""))
    return ""


def _summary_message(state: PantryAgentState, payload: dict[str, Any], fallback: str) -> str:
    """Build a concise UI summary when structured payload is available."""
    if not payload:
        return fallback

    parts: list[str] = []

    extracted_items = payload.get("extracted_items")
    if isinstance(extracted_items, list):
        parts.append(f"Identified {len(extracted_items)} pantry item(s).")

    recipes = payload.get("recipes")
    if isinstance(recipes, list):
        parts.append(f"Prepared {len(recipes)} recipe recommendation(s).")

    pantry_items = payload.get("pantry_items")
    if isinstance(pantry_items, list):
        parts.append(f"Loaded {len(pantry_items)} pantry item(s) from inventory.")

    waste_analysis = payload.get("waste_analysis")
    if isinstance(waste_analysis, list):
        parts.append(f"Computed waste risk for {len(waste_analysis)} item(s).")

    sustainability = payload.get("sustainability")
    if isinstance(sustainability, dict) and sustainability:
        parts.append("Generated sustainability insights.")

    if state.get("human_approval_required"):
        parts.append("Approval is required before applying changes.")

    errors = state.get("validation_errors", [])
    if isinstance(errors, list) and errors:
        parts.append(f"{len(errors)} validation issue(s) detected.")

    if parts:
        return " ".join(parts)

    return "Structured results are ready in payload."


def _selection_list_artifact(items: list[dict[str, Any]]) -> UIArtifact:
    return UIArtifact(
        artifact_id="ingredients-selection",
        type="selection_list",
        title="Detected Ingredients",
        description="Select ingredients to use for the next step.",
        data={
            "items": [
                {
                    "id": str(idx),
                    "label": item.get("name", "Unknown"),
                    "quantity": item.get("quantity"),
                    "unit": item.get("unit"),
                    "category": item.get("category"),
                    "selected_default": True,
                    "confidence": item.get("confidence"),
                }
                for idx, item in enumerate(items)
            ]
        },
        actions=[
            UIAction(
                action_id="confirm-selected-ingredients",
                label="Use Selected Ingredients",
                kind=ActionKind.SUBMIT_SELECTION,
                payload={"selection_key": "ingredient_ids"},
                requires_confirmation=False,
            ),
            UIAction(
                action_id="refresh-ingredient-extraction",
                label="Re-extract Ingredients",
                kind=ActionKind.REQUEST_REFRESH,
                payload={"reason": "user_requested_reextract"},
            ),
        ],
        layout=UILayout(variant="checkbox_list", group_by="category"),
        accessibility={"role": "group", "aria_label": "Detected ingredients list"},
        meta={"domain": "Vision"},
    )


def _recipe_cards_artifact(recipes: list[dict[str, Any]]) -> UIArtifact:
    cards = []
    for idx, recipe in enumerate(recipes):
        cards.append(
            {
                "id": recipe.get("id") or f"recipe-{idx}",
                "image_url": recipe.get("image_url"),
                "name": recipe.get("name"),
                "prep_time_minutes": recipe.get("prep_time_minutes"),
                "calories": recipe.get("calories"),
                "difficulty": recipe.get("difficulty"),
                "rating": recipe.get("rating"),
                "metadata": {
                    "cook_time_minutes": recipe.get("cook_time_minutes"),
                    "servings": recipe.get("servings"),
                    "cuisine": recipe.get("cuisine"),
                    "tags": recipe.get("tags", []),
                },
            }
        )

    return UIArtifact(
        artifact_id="recipe-recommendations",
        type="card_collection",
        title="Recipe Recommendations",
        description="Recipes matched to your pantry and preferences.",
        data={"cards": cards},
        actions=[
            UIAction(
                action_id="request-more-recipes",
                label="Show More",
                kind=ActionKind.REQUEST_REFRESH,
                payload={"source": "recipe_recommendations"},
            )
        ],
        layout=UILayout(
            variant="media_left_meta_right",
            density="comfortable",
            media_position="left",
        ),
        accessibility={"role": "list", "aria_label": "Recipe recommendations"},
        meta={"domain": "Recipes"},
    )


def _pantry_table_artifact(items: list[dict[str, Any]]) -> UIArtifact:
    return UIArtifact(
        artifact_id="pantry-inventory",
        type="table",
        title="Pantry Inventory",
        description="Current pantry snapshot.",
        data={
            "columns": ["name", "quantity", "unit", "category", "expiry_date"],
            "rows": items,
        },
        layout=UILayout(variant="compact"),
        accessibility={"role": "table", "aria_label": "Pantry inventory table"},
        meta={"domain": "Pantry"},
    )


def _approval_artifact(state: PantryAgentState) -> UIArtifact:
    return UIArtifact(
        artifact_id="human-approval",
        type="approval_prompt",
        title="Approval Required",
        description=state.get("approval_reason") or "This action requires your review.",
        data={
            "reason": state.get("approval_reason"),
            "items": state.get("extracted_items", []),
        },
        actions=[
            UIAction(
                action_id="approve-action",
                label="Approve",
                kind=ActionKind.REQUEST_APPROVAL,
                payload={"approved": True},
                requires_confirmation=True,
            ),
            UIAction(
                action_id="reject-action",
                label="Reject",
                kind=ActionKind.REQUEST_APPROVAL,
                payload={"approved": False},
                requires_confirmation=True,
            ),
        ],
        layout=UILayout(variant="modal"),
        accessibility={"role": "dialog", "aria_label": "Approval prompt"},
        meta={"domain": state.get("domain", "General")},
    )


def compose_response(state: PantryAgentState) -> dict[str, Any]:
    """Compose a versioned UI response envelope from current agent state."""
    messages = state.get("messages", [])
    message = _last_assistant_text(messages)
    if not message and messages and isinstance(messages[-1], AIMessage):
        message = _message_content_text(messages[-1].content)

    artifacts: list[UIArtifact] = []
    actions: list[UIAction] = []

    extracted_items = state.get("extracted_items", [])
    if extracted_items:
        artifacts.append(_selection_list_artifact(extracted_items))

    recipes = state.get("recipes", [])
    if recipes:
        artifacts.append(_recipe_cards_artifact(recipes))

    pantry_items = state.get("pantry_items", [])
    if pantry_items:
        artifacts.append(_pantry_table_artifact(pantry_items))

    if state.get("human_approval_required"):
        artifacts.append(_approval_artifact(state))

    if artifacts:
        actions.append(
            UIAction(
                action_id="continue-chat",
                label="Continue",
                kind=ActionKind.CONTINUE,
                payload={"thread_context": "preserve"},
            )
        )

    payload: dict[str, Any] = {}
    if extracted_items:
        payload["extracted_items"] = extracted_items
    if recipes:
        payload["recipes"] = recipes
    if pantry_items:
        payload["pantry_items"] = pantry_items
    if state.get("waste_analysis"):
        payload["waste_analysis"] = state.get("waste_analysis", [])
    if state.get("sustainability_data"):
        payload["sustainability"] = state.get("sustainability_data", {})
    if state.get("tool_outputs"):
        payload["tool_outputs"] = state.get("tool_outputs", [])

    response = AgentResponseEnvelope(
        thread_id="",
        message=_summary_message(state, payload, message),
        payload=payload,
        artifacts=artifacts,
        actions=actions,
        context={
            "intent": state.get("intent", ""),
            "domain": state.get("domain", ""),
        },
        trace=state.get("execution_trace", []),
        approval={
            "required": bool(state.get("human_approval_required")),
            "approved": state.get("human_approved"),
            "reason": state.get("approval_reason", ""),
        },
        errors=state.get("validation_errors", []),
    )

    return {"ui_response": response.model_dump()}
