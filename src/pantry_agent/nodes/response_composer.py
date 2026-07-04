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


def _has_recipe_details_payload(payload: Any) -> bool:
    return bool(
        isinstance(payload, dict)
        and isinstance(payload.get("recipe"), dict)
        and payload.get("recipe")
    )


def _dedupe_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _preferences_payload(memory: dict[str, Any]) -> dict[str, Any]:
    substitutions_raw = memory.get("substitutions", {})
    substitutions = (
        {str(k): str(v) for k, v in substitutions_raw.items()}
        if isinstance(substitutions_raw, dict)
        else {}
    )
    return {
        "dietary_preferences": _dedupe_string_list(memory.get("dietary_preferences", [])),
        "allergies": _dedupe_string_list(memory.get("allergies", [])),
        "favourite_recipes": _dedupe_string_list(memory.get("favourite_recipes", [])),
        "substitutions": substitutions,
    }


def _pantry_items_from_tool_outputs(tool_outputs: Any) -> tuple[list[dict[str, Any]] | None, bool]:
    """Extract pantry items from get_pantry_inventory tool outputs when available."""
    if not isinstance(tool_outputs, list):
        return None, False

    for output in tool_outputs:
        if not isinstance(output, dict):
            continue
        if output.get("tool_name") != "get_pantry_inventory":
            continue

        data = output.get("data")
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                return items, True
            return [], True

        if isinstance(data, list):
            return data, True

        return [], True

    return None, False


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

    recipe_details = payload.get("recipe_details")
    if _has_recipe_details_payload(recipe_details):
        recipe = recipe_details.get("recipe", {})
        recipe_name = recipe.get("name") or recipe.get("title") or "selected recipe"
        parts.append(f"Prepared details for {recipe_name}.")

    pantry_items = payload.get("pantry_items")
    if isinstance(pantry_items, list):
        if pantry_items:
            parts.append(f"Loaded {len(pantry_items)} pantry item(s) from inventory.")
        else:
            parts.append("No pantry items found.")
    else:
        tool_outputs = payload.get("tool_outputs")
        if isinstance(tool_outputs, list):
            for output in tool_outputs:
                if output.get("tool_name") != "get_pantry_inventory":
                    continue
                data = output.get("data", {})
                if isinstance(data, dict) and isinstance(data.get("items"), list) and len(data["items"]) == 0:
                    parts.append("No pantry items found.")
                    break

    waste_analysis = payload.get("waste_analysis")
    if isinstance(waste_analysis, list):
        parts.append(f"Computed waste risk for {len(waste_analysis)} item(s).")

    sustainability = payload.get("sustainability")
    if isinstance(sustainability, dict) and sustainability:
        parts.append("Generated sustainability insights.")

    preferences = payload.get("preferences")
    if isinstance(preferences, dict):
        prefs_count = len(preferences.get("dietary_preferences", []))
        allergy_count = len(preferences.get("allergies", []))
        parts.append(
            "Loaded saved preferences from memory"
            f" ({prefs_count} dietary preference(s), {allergy_count} allergy item(s))."
        )

    if state.get("human_approval_required"):
        parts.append("Approval is required before applying changes.")

    errors = state.get("validation_errors", [])
    if isinstance(errors, list) and errors:
        parts.append(f"{len(errors)} validation issue(s) detected.")

    tool_outputs = payload.get("tool_outputs")
    if isinstance(tool_outputs, list):
        added_items = 0
        for output in tool_outputs:
            if not isinstance(output, dict):
                continue

            tool_name = output.get("tool_name")
            data = output.get("data", {})
            if not isinstance(data, dict) or data.get("error"):
                continue

            if tool_name == "add_pantry_item":
                added_items += 1
                continue

            if tool_name == "extract_and_save_pantry_items":
                saved_items = data.get("saved_items")
                if isinstance(saved_items, list):
                    added_items += len(saved_items)

        if added_items > 0:
            parts.append("Items added successfully.")

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


def _recipe_card_data(recipe: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": recipe.get("id") or f"recipe-{index}",
        "image_url": recipe.get("image_url") or recipe.get("image"),
        "name": recipe.get("name") or recipe.get("title"),
        "prep_time_minutes": recipe.get("prep_time_minutes") or recipe.get("total_time"),
        "calories": recipe.get("calories"),
        "difficulty": recipe.get("difficulty"),
        "rating": recipe.get("rating"),
        "metadata": {
            "cook_time_minutes": recipe.get("cook_time_minutes"),
            "servings": recipe.get("servings"),
            "cuisine": recipe.get("cuisine"),
            "tags": recipe.get("tags", []),
            "url": recipe.get("url"),
            "protein": recipe.get("protein"),
            "hybrid_score": recipe.get("hybrid_score"),
        },
    }


def _recipe_cards_artifact(recipes: list[dict[str, Any]]) -> UIArtifact:
    cards = [_recipe_card_data(recipe, idx) for idx, recipe in enumerate(recipes)]

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


def _recipe_details_artifact(recipe_details: dict[str, Any]) -> UIArtifact:
    recipe = recipe_details.get("recipe", {}) if isinstance(recipe_details, dict) else {}
    analysis = {
        "pantry_coverage_pct": recipe_details.get("pantry_coverage_pct"),
        "available_ingredients": recipe_details.get("available_ingredients", []),
        "missing_ingredients": recipe_details.get("missing_ingredients", []),
        "substitutions": recipe_details.get("substitutions", {}),
        "pantry_matches": recipe_details.get("pantry_matches", []),
        "shopping_list": recipe_details.get("shopping_list", []),
        "message": recipe_details.get("message"),
    }

    recipe_name = recipe.get("name") or recipe.get("title") or "Recipe details"
    actions: list[UIAction] = []
    if recipe.get("url"):
        actions.append(
            UIAction(
                action_id="open-recipe-source",
                label="Open Recipe",
                kind=ActionKind.OPEN_DETAILS,
                payload={"url": recipe.get("url"), "recipe_id": recipe.get("id")},
                requires_confirmation=False,
            )
        )

    return UIArtifact(
        artifact_id="recipe-details",
        type="detail_view",
        title=recipe_name,
        description=recipe.get("description") or analysis.get("message"),
        data={"recipe": recipe, "analysis": analysis},
        actions=actions,
        layout=UILayout(variant="detail", density="comfortable"),
        accessibility={"role": "article", "aria_label": f"Recipe details for {recipe_name}"},
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


def _preferences_artifact(preferences: dict[str, Any]) -> UIArtifact:
    substitutions = preferences.get("substitutions", {})
    substitution_rows = [
        {"ingredient": key, "substitute": value}
        for key, value in substitutions.items()
    ]

    return UIArtifact(
        artifact_id="user-preferences",
        type="detail_view",
        title="Saved Preferences",
        description="Preferences loaded from your long-term memory.",
        data={
            "dietary_preferences": preferences.get("dietary_preferences", []),
            "allergies": preferences.get("allergies", []),
            "favourite_recipes": preferences.get("favourite_recipes", []),
            "substitutions": substitution_rows,
        },
        layout=UILayout(variant="detail", density="comfortable"),
        accessibility={"role": "region", "aria_label": "Saved dietary preferences"},
        meta={"domain": "General", "intent": "get_preferences"},
    )


def compose_response(state: PantryAgentState) -> dict[str, Any]:
    """Compose a versioned UI response envelope from current agent state."""
    messages = state.get("messages", [])
    message = _last_assistant_text(messages)
    if not message and messages and isinstance(messages[-1], AIMessage):
        message = _message_content_text(messages[-1].content)

    artifacts: list[UIArtifact] = []
    actions: list[UIAction] = []
    intent = state.get("intent", "")
    tool_outputs = state.get("tool_outputs", [])

    extracted_items = state.get("extracted_items", [])
    if extracted_items:
        artifacts.append(_selection_list_artifact(extracted_items))

    recipes = state.get("recipes", [])
    if recipes:
        artifacts.append(_recipe_cards_artifact(recipes))

    recipe_details = state.get("recipe_details", {})
    if _has_recipe_details_payload(recipe_details):
        artifacts.append(_recipe_details_artifact(recipe_details))

    pantry_items_raw = state.get("pantry_items")
    pantry_items: list[dict[str, Any]] = pantry_items_raw if isinstance(pantry_items_raw, list) else []
    derived_pantry_items, pantry_lookup_called = _pantry_items_from_tool_outputs(tool_outputs)
    if derived_pantry_items is not None:
        pantry_items = derived_pantry_items

    if pantry_items or pantry_lookup_called or intent == "get_pantry":
        artifacts.append(_pantry_table_artifact(pantry_items))

    preferences = _preferences_payload(state.get("memory", {})) if intent == "get_preferences" else {}
    if preferences:
        artifacts.append(_preferences_artifact(preferences))

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
    if _has_recipe_details_payload(state.get("recipe_details")):
        payload["recipe_details"] = state.get("recipe_details", {})
    if pantry_items or pantry_lookup_called or intent == "get_pantry":
        payload["pantry_items"] = pantry_items
    if preferences:
        payload["preferences"] = preferences
    if state.get("waste_analysis"):
        payload["waste_analysis"] = state.get("waste_analysis", [])
    if state.get("sustainability_data"):
        payload["sustainability"] = state.get("sustainability_data", {})
    if tool_outputs:
        payload["tool_outputs"] = tool_outputs

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
