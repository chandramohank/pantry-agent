"""
Food Waste Reduction Tools
==========================
Four tools covering waste analysis, recipe-based waste reduction,
dashboard analytics, and top-risk item identification.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_get, safe_api_call

logger = logging.getLogger(__name__)


# ── Input schemas ─────────────────────────────────────────────────────────────

class AnalyzeWasteInput(BaseModel):
    risk_level_filter: str | None = Field(
        default=None,
        description="Filter results by risk level: high | medium | low",
    )
    category: str | None = Field(
        default=None,
        description="Focus analysis on a specific food category.",
    )


class TopRiskInput(BaseModel):
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top-risk items to return (default: 10).",
    )


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool(args_schema=AnalyzeWasteInput)
def analyze_food_waste(
    risk_level_filter: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Analyse the user's pantry to identify ingredients at risk of being wasted.

    PURPOSE
    -------
    Scans every pantry item and calculates a waste risk score based on
    expiry dates, quantity, typical usage patterns, and spoilage rates.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "What is expiring soon?"
    - Asks "Which foods are about to go bad?"
    - Asks "Show me waste risks in my fridge"
    - Asks "What should I use up this week?"
    - Asks "Which items will spoil if I don't use them?"
    - Wants a general overview of pantry health / waste status

    WHEN NOT TO USE
    ---------------
    - For recipe suggestions to USE expiring items → use recommend_waste_reduction
    - For overall analytics dashboard → use waste_dashboard
    - For just the highest risk items → use top_risk_items

    RETURNS
    -------
    {
      "waste_items": [
        {
          "pantry_item": {"name": "spinach", "quantity": 200, "unit": "g", ...},
          "waste_score": 0.92,           // 0.0 (safe) to 1.0 (certain waste)
          "risk_level": "high",
          "days_until_expiry": 1,
          "expiry_prediction": "2025-07-04",
          "recommended_action": "Use today in a salad or smoothie."
        },
        ...
      ],
      "total_items_analysed": 38,
      "high_risk_count": 3,
      "medium_risk_count": 7,
      "low_risk_count": 28,
      "estimated_waste_value": 12.50
    }
    """
    params: dict[str, Any] = {}
    if risk_level_filter:
        params["risk_level"] = risk_level_filter
    if category:
        params["category"] = category

    result = safe_api_call(api_get, "/api/ai/waste-analysis", params or None)
    logger.info(
        "Waste analysis: %d high, %d medium, %d low risk",
        result.get("high_risk_count", 0),
        result.get("medium_risk_count", 0),
        result.get("low_risk_count", 0),
    )
    return result


@tool
def recommend_waste_reduction() -> dict[str, Any]:
    """
    Recommend recipes and actions that specifically use ingredients nearing expiration.

    PURPOSE
    -------
    Reduces food waste by suggesting recipes that prioritise the pantry items
    with the highest spoilage risk, ensuring they are consumed before going bad.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "How can I avoid food waste?"
    - Asks "What should I cook to use up expiring food?"
    - Asks "Give me recipes for the vegetables that are about to expire"
    - Asks "Help me use up what's in my fridge before it goes bad"
    - Asks "Cook before they expire"
    - After showing waste analysis results, to provide actionable next steps

    WHEN NOT TO USE
    ---------------
    - For general recipe discovery → use recommend_recipes
    - For waste statistics only → use analyze_food_waste or waste_dashboard
    - For cooking instructions on a specific dish → use cooking_copilot

    RETURNS
    -------
    {
      "recipes": [
        {
          "name": "Spinach and Feta Omelette",
          "description": "Uses your expiring spinach and feta cheese...",
          "ingredients": [...],
          ...
        }
      ],
      "priority_items": ["spinach (expires tomorrow)", "Greek yogurt (2 days)"],
      "tips": [
        "Blanch and freeze the spinach if you can't cook it today.",
        "The yogurt can be frozen in portions."
      ]
    }
    """
    result = safe_api_call(api_get, "/api/ai/waste-reduction")
    logger.info("Waste reduction: %d recipes returned", len(result.get("recipes", [])))
    return result


@tool
def waste_dashboard() -> dict[str, Any]:
    """
    Get a high-level summary of the user's pantry waste analytics.

    PURPOSE
    -------
    Provides a bird's-eye view of pantry health, showing aggregate counts
    of items at each risk level and an overall waste score.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "Show me a waste summary"
    - Asks "How is my pantry doing in terms of waste?"
    - Asks "Give me a food waste overview"
    - Wants a dashboard / summary view rather than item-level detail
    - During routine check-ins / morning pantry briefings

    WHEN NOT TO USE
    ---------------
    - For item-level waste detail → use analyze_food_waste
    - For the specific worst offenders → use top_risk_items
    - For waste-reducing recipe suggestions → use recommend_waste_reduction

    RETURNS
    -------
    {
      "total_items": 42,
      "high_risk": 3,
      "medium_risk": 7,
      "low_risk": 28,
      "no_risk": 4,
      "waste_score_avg": 0.18,
      "last_updated": "2025-07-03T08:00:00Z"
    }
    """
    result = safe_api_call(api_get, "/api/dashboard/waste")
    logger.info(
        "Waste dashboard: %d total items, avg score=%.2f",
        result.get("total_items", 0),
        result.get("waste_score_avg", 0),
    )
    return result


@tool(args_schema=TopRiskInput)
def top_risk_items(limit: int = 10) -> dict[str, Any]:
    """
    Retrieve the pantry items with the highest spoilage risk right now.

    PURPOSE
    -------
    Returns an ordered list of the most urgent items in the pantry –
    those that need to be used or discarded soonest.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "What should I use today / urgently?"
    - Asks "Show me the most at-risk items"
    - Asks "What will expire first?"
    - Wants a quick prioritised action list
    - Is doing a quick morning pantry check

    WHEN NOT TO USE
    ---------------
    - For the full waste analysis → use analyze_food_waste
    - For summary counts only → use waste_dashboard
    - For recipes using those items → use recommend_waste_reduction

    RETURNS
    -------
    {
      "items": [
        {
          "item": {"name": "spinach", "quantity": 200, "unit": "g", ...},
          "risk_level": "high",
          "waste_score": 0.95,
          "days_until_expiry": 0
        },
        ...
      ],
      "total": 10
    }
    """
    params: dict[str, Any] = {"limit": limit}
    result = safe_api_call(api_get, "/api/dashboard/waste/top-risk", params)
    logger.info("Top risk items: %d returned", result.get("total", 0))
    return result
