"""Tests for tool definitions and registry."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from pantry_agent.tools.registry import (
    DOMAIN_TOOLS,
    TOOL_BY_NAME,
    get_all_tools,
    get_tool_names,
    get_tools_for_domain,
)


# ── Registry tests ────────────────────────────────────────────────────────────

def test_all_tools_returns_list():
    tools = get_all_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0


def test_tool_names_are_unique():
    names = get_tool_names()
    assert len(names) == len(set(names)), "Duplicate tool names detected"


def test_domain_tools_coverage():
    expected_domains = {"Vision", "Pantry", "Recipes", "Diet", "Cooking", "Waste", "Sustainability", "General"}
    assert set(DOMAIN_TOOLS.keys()) == expected_domains


def test_get_tools_for_unknown_domain_falls_back_to_general():
    tools = get_tools_for_domain("Unknown")
    assert tools == DOMAIN_TOOLS["General"]


def test_tool_by_name_lookup():
    for name in get_tool_names():
        assert name in TOOL_BY_NAME, f"Tool '{name}' not in TOOL_BY_NAME"


def test_vision_tools_in_vision_domain():
    vision_tool_names = {t.name for t in DOMAIN_TOOLS["Vision"]}
    assert "extract_ingredients_from_image" in vision_tool_names
    assert "ask_about_uploaded_image" in vision_tool_names


# ── Tool invocation tests (mocked API) ───────────────────────────────────────

def _mock_api_response(data: dict[str, Any]) -> Any:
    """Return a mock function that returns `data`."""
    def _fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return data
    return _fn


def test_get_pantry_inventory_calls_api(sample_pantry_items):
    from pantry_agent.tools.pantry import get_pantry_inventory

    mock_response = {"items": sample_pantry_items, "total": len(sample_pantry_items)}
    with patch("pantry_agent.tools.pantry.safe_api_call", return_value=mock_response):
        result = get_pantry_inventory.invoke({})
    assert result["total"] == len(sample_pantry_items)
    assert len(result["items"]) == len(sample_pantry_items)


def test_add_pantry_item_calls_api():
    from pantry_agent.tools.pantry import add_pantry_item

    mock_response = {"id": "new-1", "name": "butter", "quantity": 250.0, "unit": "g"}
    with patch("pantry_agent.tools.pantry.safe_api_call", return_value=mock_response):
        result = add_pantry_item.invoke({"name": "butter", "quantity": 250.0, "unit": "g"})
    assert result["name"] == "butter"


def test_extract_pantry_items_from_text():
    from pantry_agent.tools.ai_extract import extract_pantry_items_from_text

    mock_response = {
        "extracted_items": [
            {"name": "milk", "quantity": 1.0, "unit": "litre"},
            {"name": "eggs", "quantity": 6.0, "unit": "pieces"},
        ],
        "confidence": 0.95,
    }
    with patch("pantry_agent.tools.ai_extract.safe_api_call", return_value=mock_response):
        result = extract_pantry_items_from_text.invoke({"text": "I bought milk and 6 eggs"})
    assert len(result["extracted_items"]) == 2


def test_recommend_recipes_returns_recipes(sample_recipes):
    from pantry_agent.tools.recipes import recommend_recipes

    mock_response = {"recipes": sample_recipes, "pantry_coverage_pct": 90.0}
    with patch("pantry_agent.tools.recipes.safe_api_call", return_value=mock_response):
        result = recommend_recipes.invoke({})
    assert len(result["recipes"]) == 1
    assert result["recipes"][0]["name"] == "Spinach Omelette"


def test_sustainability_insights_tool():
    from pantry_agent.tools.sustainability import sustainability_insights

    mock_response = {
        "insights": [{"category": "Food Waste", "insight": "...", "recommendation": "...", "impact_score": 0.7}],
        "recommendations": ["Buy seasonal produce"],
        "actions": ["Compost peels"],
        "summary": "Score: 72/100",
        "overall_score": 72.0,
    }
    with patch("pantry_agent.tools.sustainability.safe_api_call", return_value=mock_response):
        result = sustainability_insights.invoke({})
    assert result["overall_score"] == 72.0


def test_extract_ingredients_requires_image_or_url():
    from pantry_agent.tools.vision import extract_ingredients_from_image

    # No image provided – should return error
    result = extract_ingredients_from_image.invoke({"auto_save": False})
    assert result.get("error") is True


def test_analyze_food_waste():
    from pantry_agent.tools.waste import analyze_food_waste

    mock_response = {
        "waste_items": [],
        "total_items_analysed": 10,
        "high_risk_count": 2,
        "medium_risk_count": 3,
        "low_risk_count": 5,
    }
    with patch("pantry_agent.tools.waste.safe_api_call", return_value=mock_response):
        result = analyze_food_waste.invoke({})
    assert result["high_risk_count"] == 2
