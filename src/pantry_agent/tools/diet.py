"""
Diet Planner Tool
=================
Generate personalised multi-day meal plans using pantry contents.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_post, safe_api_call

logger = logging.getLogger(__name__)


class CreateDietPlanInput(BaseModel):
    diet_type: str = Field(
        default="standard",
        description=(
            "Dietary pattern: standard | vegetarian | vegan | keto | paleo | "
            "gluten_free | high_protein | low_carb | mediterranean"
        ),
    )
    days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Number of days to plan meals for (1–30).",
    )
    calories_per_day: int | None = Field(
        default=None,
        ge=500,
        le=5000,
        description="Target daily calorie intake. Omit to let the API decide.",
    )
    allergies: list[str] = Field(
        default_factory=list,
        description=(
            "List of allergens to avoid, e.g. ['nuts', 'gluten', 'shellfish', 'dairy']."
        ),
    )
    preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Positive food preferences or cuisines, e.g. ['Italian', 'spicy', 'quick meals']."
        ),
    )
    use_pantry_items: bool = Field(
        default=True,
        description="Incorporate currently available pantry ingredients into the plan.",
    )


@tool(args_schema=CreateDietPlanInput)
def create_diet_plan(
    diet_type: str = "standard",
    days: int = 7,
    calories_per_day: int | None = None,
    allergies: list[str] | None = None,
    preferences: list[str] | None = None,
    use_pantry_items: bool = True,
) -> dict[str, Any]:
    """
    Generate a personalised multi-day meal plan tailored to dietary preferences
    and available pantry ingredients.

    PURPOSE
    -------
    Creates a structured day-by-day meal plan (breakfast, lunch, dinner, snacks)
    optimised for the chosen diet type, user preferences, and pantry contents.
    Also produces a shopping list for missing ingredients.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "Create a keto meal plan for the week"
    - Asks "Plan vegetarian meals for 5 days"
    - Asks "Make me a high-protein diet plan"
    - Asks "What should I eat this week?"
    - Asks "Generate a meal plan using what I have"
    - Asks about structured weekly / monthly diet planning

    WHEN NOT TO USE
    ---------------
    - For a single recipe suggestion → use recommend_recipes
    - For cooking instructions → use cooking_copilot
    - For checking pantry contents → use get_pantry_inventory

    RETURNS
    -------
    {
      "meal_plans": [
        {
          "day": 1,
          "breakfast": { "name": "Greek yogurt bowl", ... },
          "lunch": { "name": "Chicken Caesar salad", ... },
          "dinner": { "name": "Grilled salmon with vegetables", ... },
          "snacks": [...]
        },
        ...
      ],
      "available_ingredients": ["chicken breast", "eggs", "spinach", ...],
      "missing_ingredients": ["salmon fillet", "Greek yogurt"],
      "substitutions": {"salmon fillet": "canned tuna"},
      "shopping_list": [{"name": "salmon fillet", "quantity": 4, "unit": "pieces"}],
      "summary": "7-day keto plan using 85% of your pantry items."
    }

    DIETARY TYPES SUPPORTED
    -----------------------
    standard | vegetarian | vegan | keto | paleo | gluten_free |
    high_protein | low_carb | mediterranean
    """
    payload: dict[str, Any] = {
        "diet_type": diet_type,
        "days": days,
        "use_pantry_items": use_pantry_items,
        "allergies": allergies or [],
        "preferences": preferences or [],
    }
    if calories_per_day is not None:
        payload["calories_per_day"] = calories_per_day

    result = safe_api_call(api_post, "/api/ai/diet-planner", payload)
    logger.info(
        "Diet plan created: %d days, type=%s", len(result.get("meal_plans", [])), diet_type
    )
    return result
