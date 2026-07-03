"""
Sustainability Insights Tool
============================
Generate environmental sustainability recommendations based on pantry contents.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from ..api_client import api_get, safe_api_call

logger = logging.getLogger(__name__)


@tool
def sustainability_insights() -> dict[str, Any]:
    """
    Generate personalised sustainability and environmental impact insights
    based on the user's pantry contents and food consumption patterns.

    PURPOSE
    -------
    Analyses the user's pantry to provide actionable recommendations for
    reducing food waste, carbon footprint, and environmental impact.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "How sustainable is my pantry / diet?"
    - Asks "What is my carbon footprint from food?"
    - Asks "How can I reduce my environmental impact?"
    - Asks "Give me sustainability tips"
    - Asks "How eco-friendly is my shopping?"
    - Asks "Help me reduce food waste and emissions"
    - Asks about the environmental impact of specific food choices
    - Wants to improve their sustainability score

    WHEN NOT TO USE
    ---------------
    - For food waste specifics → use analyze_food_waste or waste_dashboard
    - For recipe suggestions → use recommend_recipes or recommend_waste_reduction
    - For diet planning → use create_diet_plan

    RETURNS
    -------
    {
      "insights": [
        {
          "category": "Food Waste",
          "insight": "You waste approximately 20% of fresh produce.",
          "recommendation": "Plan meals around expiring items first.",
          "impact_score": 0.7
        },
        {
          "category": "Carbon Footprint",
          "insight": "Your meat consumption is above average.",
          "recommendation": "Try 2 meat-free days per week.",
          "impact_score": 0.85
        }
      ],
      "recommendations": [
        "Switch to seasonal produce to reduce transport emissions.",
        "Buy in bulk to reduce packaging waste."
      ],
      "actions": [
        "Add a meat-free Monday to your meal plan.",
        "Compost vegetable peelings instead of discarding them."
      ],
      "summary": "Your pantry scores 62/100 for sustainability. Small changes can boost this significantly.",
      "overall_score": 62.0,
      "carbon_footprint_estimate": "~4.2 kg CO2e per week"
    }
    """
    result = safe_api_call(api_get, "/api/ai/sustainability-insights")
    logger.info(
        "Sustainability insights: %d insights, score=%s",
        len(result.get("insights", [])),
        result.get("overall_score", "N/A"),
    )
    return result
