"""
Tool Registry
=============
Central registry mapping domain labels to tool sets.
Provides helpers for domain-filtered tool selection and full tool access.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from .ai_extract import extract_and_save_pantry_items, extract_pantry_items_from_text
from .cooking import cooking_copilot
from .diet import create_diet_plan
from .pantry import add_pantry_item, get_pantry_inventory
from .recipes import recommend_recipes, recipe_search_tool
from .sustainability import sustainability_insights
from .vision import ask_about_uploaded_image, extract_ingredients_from_image
from .waste import analyze_food_waste, recommend_waste_reduction, top_risk_items, waste_dashboard

# ── Domain → tool subsets ─────────────────────────────────────────────────────
# Tools are listed in priority order within each domain.

DOMAIN_TOOLS: dict[str, list[BaseTool]] = {
    "Vision": [
        extract_ingredients_from_image,
        ask_about_uploaded_image,
    ],
    "Pantry": [
        get_pantry_inventory,
        add_pantry_item,
        extract_pantry_items_from_text,
        extract_and_save_pantry_items,
    ],
    "Recipes": [
        recommend_recipes,
        recipe_search_tool,
        get_pantry_inventory,
    ],
    "Diet": [
        create_diet_plan,
        get_pantry_inventory,
    ],
    "Cooking": [
        cooking_copilot,
        get_pantry_inventory,
    ],
    "Waste": [
        analyze_food_waste,
        recommend_waste_reduction,
        waste_dashboard,
        top_risk_items,
        get_pantry_inventory,
    ],
    "Sustainability": [
        sustainability_insights,
        analyze_food_waste,
        waste_dashboard,
    ],
    "General": [
        # All tools available for unclassified / multi-domain queries.
        get_pantry_inventory,
        recommend_recipes,
        recipe_search_tool,
        cooking_copilot,
        analyze_food_waste,
        recommend_waste_reduction,
        sustainability_insights,
        create_diet_plan,
        add_pantry_item,
        extract_pantry_items_from_text,
        extract_and_save_pantry_items,
        waste_dashboard,
        top_risk_items,
        extract_ingredients_from_image,
        ask_about_uploaded_image,
    ],
}

# Vision tools always receive highest priority when images are present.
VISION_PRIORITY_TOOLS: list[BaseTool] = [
    extract_ingredients_from_image,
    ask_about_uploaded_image,
    get_pantry_inventory,
    recommend_recipes,
    cooking_copilot,
    add_pantry_item,
    extract_and_save_pantry_items,
]


def get_tools_for_domain(domain: str) -> list[BaseTool]:
    """Return the tool subset for a given domain label."""
    return DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS["General"])


def get_all_tools() -> list[BaseTool]:
    """Return every tool – used when building the ToolNode."""
    return DOMAIN_TOOLS["General"]


def get_tool_names() -> list[str]:
    """Return the names of all registered tools."""
    return [t.name for t in get_all_tools()]


# Tool name → tool object lookup for fast access
TOOL_BY_NAME: dict[str, BaseTool] = {t.name: t for t in get_all_tools()}
