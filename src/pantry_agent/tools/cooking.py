"""
Cooking Copilot Tool
====================
Conversational kitchen assistant for cooking questions, substitutions,
techniques, and real-time cooking guidance.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_post, safe_api_call

logger = logging.getLogger(__name__)


class CookingCopilotInput(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description=(
            "The cooking question or request. Be specific for best results. "
            "E.g. 'How long should I bake lasagna at 180°C?' or "
            "'What can I substitute for butter in chocolate cake?'"
        ),
    )
    recipe_name: str | None = Field(
        default=None,
        description=(
            "Recipe name the question refers to. Use when asking for steps, timings, "
            "or substitutions for a specific dish."
        ),
    )
    context: str | None = Field(
        default=None,
        description=(
            "Additional context to improve the answer: recipe name, ingredients "
            "currently available, cooking equipment, skill level, or dietary constraints."
        ),
    )
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Previous Q&A pairs in this cooking session to maintain continuity. "
            "Format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]"
        ),
    )


@tool(args_schema=CookingCopilotInput)
def cooking_copilot(
    question: str,
    recipe_name: str | None = None,
    context: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Answer cooking questions, provide substitutions, and give step-by-step guidance
    as a personal kitchen assistant.

    PURPOSE
    -------
    The cooking copilot is your conversational companion in the kitchen.
    It handles a wide range of culinary questions using AI-powered knowledge.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks a HOW-TO cooking question: "How do I make béchamel sauce?"
    - Asks about cooking times / temperatures: "How long to bake chicken at 200°C?"
    - Asks about ingredient substitutions: "What can I use instead of eggs?"
    - Asks about cooking techniques: "How do I julienne carrots?"
    - Asks about food safety: "Is it safe to refreeze thawed chicken?"
    - Asks "Can I use olive oil instead of vegetable oil?"
    - Asks "What if I don't have fresh herbs, can I use dried?"
    - Is actively cooking and needs real-time guidance
    - Asks about recipe modifications for dietary restrictions

    WHEN NOT TO USE
    ---------------
    - For recipe RECOMMENDATIONS based on pantry → use recommend_recipes
    - For WEEKLY meal planning → use create_diet_plan
    - For scanning images → use extract_ingredients_from_image
    - For pantry inventory questions → use get_pantry_inventory

    RETURNS
    -------
    {
      "answer": "Bake lasagna at 180°C (fan) for 45–55 minutes, covered with foil for the first 30 minutes...",
      "tips": [
        "Let it rest 10 minutes before cutting for cleaner slices.",
        "The internal temperature should reach 75°C."
      ],
      "substitutions": {
        "butter": "coconut oil or margarine in equal quantities"
      },
      "related_recipes": ["moussaka", "baked ziti"]
    }

    MULTI-TURN CONVERSATION
    -----------------------
    Pass conversation_history to maintain cooking session continuity.
    The context field is especially useful for referencing a specific recipe.
    """
    resolved_recipe_name = (recipe_name or "").strip()
    if not resolved_recipe_name and context:
        # Common pattern: "recipe name: X" inside context text.
        match = re.search(r"recipe\s*name\s*:\s*([^\n,.;]+)", context, flags=re.IGNORECASE)
        if match:
            resolved_recipe_name = match.group(1).strip()
    if not resolved_recipe_name:
        resolved_recipe_name = "general"

    payload: dict[str, Any] = {
        "question": question,
        "recipeName": resolved_recipe_name,
        "conversation_history": conversation_history or [],
    }
    if context:
        payload["context"] = context

    result = safe_api_call(api_post, "/api/ai/cooking-copilot", payload)
    if result.get("error"):
        logger.warning("Cooking copilot API failed for recipeName=%r", resolved_recipe_name)
        return {
            "error": True,
            "message": "Cooking copilot is temporarily unavailable. Please try again.",
            "details": result.get("message"),
        }
    logger.info("Cooking copilot answered: %s", question[:80])
    return result
