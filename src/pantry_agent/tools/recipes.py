"""
Recipe Intelligence Tool
========================
Recommend recipes that can be cooked using the user's current pantry contents.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_get, safe_api_call

logger = logging.getLogger(__name__)


class RecommendRecipesInput(BaseModel):
    cuisine: str | None = Field(
        default=None,
        description=(
            "Filter to a specific cuisine style: Italian | Mexican | Asian | "
            "Indian | French | Mediterranean | American | etc."
        ),
    )
    max_missing_ingredients: int | None = Field(
        default=None,
        description=(
            "Maximum number of ingredients allowed to be missing from the pantry. "
            "0 = only show recipes the user can cook right now."
        ),
        ge=0,
        le=20,
    )
    meal_type: str | None = Field(
        default=None,
        description="breakfast | lunch | dinner | snack | dessert",
    )
    max_cook_time_minutes: int | None = Field(
        default=None,
        description="Return only recipes that take ≤ this many minutes to cook.",
        ge=5,
        le=480,
    )


@tool(args_schema=RecommendRecipesInput)
def recommend_recipes(
    cuisine: str | None = None,
    max_missing_ingredients: int | None = None,
    meal_type: str | None = None,
    max_cook_time_minutes: int | None = None,
) -> dict[str, Any]:
    """
    Recommend recipes based on what is currently in the user's pantry.

    PURPOSE
    -------
    Queries the Pantry API for recipe recommendations personalised to the
    user's available ingredients. Recipes are ranked by pantry coverage
    (highest coverage first).

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "What can I cook tonight?"
    - Asks "Suggest a recipe"
    - Asks "Recommend dinner / lunch / breakfast ideas"
    - Asks "What recipes can I make with what I have?"
    - Asks "Give me a quick meal idea"
    - Asks "What can I make without going shopping?"

    WHEN NOT TO USE
    ---------------
    - For detailed cooking instructions → use cooking_copilot
    - For weekly meal planning → use create_diet_plan
    - For waste reduction (using expiring items) → use recommend_waste_reduction

    RETURNS
    -------
    {
      "recipes": [
        {
          "name": "Spaghetti Carbonara",
          "description": "...",
          "ingredients": [...],
          "instructions": [...],
          "prep_time_minutes": 10,
          "cook_time_minutes": 20,
          "servings": 2,
          "cuisine": "Italian",
          "tags": ["pasta", "quick"]
        }
      ],
      "pantry_coverage_pct": 85.0,
      "message": "Found 8 recipes you can cook now."
    }

    DEPENDENCIES
    ------------
    For best results, call get_pantry_inventory first so the user knows
    what they have. The API uses the authenticated user's pantry automatically.
    """
    params: dict[str, Any] = {}
    if cuisine:
        params["cuisine"] = cuisine
    if max_missing_ingredients is not None:
        params["max_missing_ingredients"] = max_missing_ingredients
    if meal_type:
        params["meal_type"] = meal_type
    if max_cook_time_minutes is not None:
        params["max_cook_time_minutes"] = max_cook_time_minutes

    result = safe_api_call(api_get, "/api/ai/recipes", params or None)
    logger.info("Recipe recommendations: %d returned", len(result.get("recipes", [])))
    return result
