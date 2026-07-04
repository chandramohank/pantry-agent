"""Tests for tool definitions and registry."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

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


def test_recommend_recipes_returns_recipes(sample_recipes, sample_pantry_items):
    from pantry_agent.tools.recipes import recommend_recipes

    # Mock recipe_search_tool to return sample recipes
    sample_recipes_with_details = [
        {
            **recipe,
            "title": recipe.get("name"),
            "url": "https://example.com/recipe1",
            "image": "https://example.com/image.jpg",
            "total_time": 10,
            "calories": 250.0,
            "protein": 15.0,
            "cuisine": "Italian",
        }
        for recipe in sample_recipes
    ]
    
    mock_search_response = {"recipes": sample_recipes_with_details, "total_found": 1, "query": "recipe recommendation", "execution_time_ms": 100.0}
    mock_pantry_response = {"items": sample_pantry_items, "total": len(sample_pantry_items)}
    
    with (
        patch("pantry_agent.tools.recipes.recipe_search_tool") as mock_search,
        patch("pantry_agent.tools.pantry.get_pantry_inventory") as mock_pantry,
    ):
        mock_search.invoke.return_value = mock_search_response
        mock_pantry.invoke.return_value = mock_pantry_response
        
        result = recommend_recipes.invoke({})
    
    assert len(result["recipes"]) == 1
    assert result["recipes"][0]["name"] == "Spinach Omelette"
    assert result["recipes"][0]["pantry_coverage_pct"] == 100.0  # All ingredients available


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


# ── Recipe Search Tool Tests ──────────────────────────────────────────────────


def test_recipe_search_tool_in_registry():
    """Verify recipe_search_tool is registered in Recipes domain."""
    from pantry_agent.tools.registry import DOMAIN_TOOLS, TOOL_BY_NAME

    recipe_tools = {t.name for t in DOMAIN_TOOLS["Recipes"]}
    assert "recipe_search_tool" in recipe_tools, "recipe_search_tool not in Recipes domain"
    assert "recipe_search_tool" in TOOL_BY_NAME, "recipe_search_tool not in TOOL_BY_NAME"


def test_recipe_search_tool_in_general_domain():
    """Verify recipe_search_tool is also in General domain."""
    from pantry_agent.tools.registry import DOMAIN_TOOLS

    general_tools = {t.name for t in DOMAIN_TOOLS["General"]}
    assert "recipe_search_tool" in general_tools, "recipe_search_tool not in General domain"


def test_get_recipe_details_in_registry():
    """Verify get_recipe_details is registered in Recipes and General domains."""
    from pantry_agent.tools.registry import DOMAIN_TOOLS, TOOL_BY_NAME

    recipe_tools = {t.name for t in DOMAIN_TOOLS["Recipes"]}
    general_tools = {t.name for t in DOMAIN_TOOLS["General"]}

    assert "get_recipe_details" in recipe_tools, "get_recipe_details not in Recipes domain"
    assert "get_recipe_details" in general_tools, "get_recipe_details not in General domain"
    assert "get_recipe_details" in TOOL_BY_NAME, "get_recipe_details not in TOOL_BY_NAME"


def test_recipe_search_tool_has_correct_schema():
    """Verify recipe_search_tool has correct input schema."""
    from pantry_agent.tools.recipes import recipe_search_tool
    from pantry_agent.models.schemas import RecipeSearchInput

    assert recipe_search_tool.args_schema is RecipeSearchInput
    assert "query" in recipe_search_tool.args_schema.model_fields
    assert "filters" in recipe_search_tool.args_schema.model_fields


def test_get_recipe_details_has_correct_schema():
    """Verify get_recipe_details has correct input schema."""
    from pantry_agent.models.schemas import RecipeDetailResponse
    from pantry_agent.tools.recipes import RecipeDetailsInput, get_recipe_details

    assert get_recipe_details.args_schema is RecipeDetailsInput
    assert "recipe" in get_recipe_details.args_schema.model_fields
    assert "include_pantry_analysis" in get_recipe_details.args_schema.model_fields
    assert RecipeDetailResponse.model_fields["recipe"].annotation is not None


def test_recipe_search_tool_missing_azure_config():
    """Test graceful fallback when Azure Search is not configured."""
    from pantry_agent.tools.recipes import recipe_search_tool

    with patch("pantry_agent.tools.recipes.settings") as mock_settings:
        mock_settings.azure_search_endpoint = ""
        mock_settings.azure_search_key = ""

        result = recipe_search_tool.invoke({"query": "beef stew"})

        assert result["recipes"] == []
        assert result["total_found"] == 0
        assert "error" in result
        assert "not configured" in result["error"]


def test_recipe_search_tool_embedding_generation_success():
    """Test successful embedding generation."""
    from pantry_agent.tools.recipes import _generate_embedding

    mock_embedding = [0.1, 0.2, 0.3] + [0.0] * 1533  # 1536 total dimensions

    with patch("pantry_agent.tools.recipes.AzureOpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.embeddings.create.return_value.data = [type("obj", (object,), {"embedding": mock_embedding})()]

        result = _generate_embedding("hearty beef stew")

        assert isinstance(result, list)
        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once()


def test_recipe_search_tool_build_odata_filter_simple():
    """Test OData filter building with simple numeric filters."""
    from pantry_agent.tools.recipes import _build_odata_filter

    filters = {"max_time": 30, "min_protein": 10}
    result = _build_odata_filter(filters)

    assert "total_time le 30" in result
    assert "protein ge 10" in result
    assert " and " in result


def test_recipe_search_tool_build_odata_filter_complex():
    """Test OData filter building with multiple complex filters."""
    from pantry_agent.tools.recipes import _build_odata_filter

    filters = {
        "max_time": 45,
        "min_protein": 15,
        "max_calories": 500,
        "max_sodium": 1000,
        "author": "Jamie Oliver",
    }
    result = _build_odata_filter(filters)

    assert "total_time le 45" in result
    assert "protein ge 15" in result
    assert "calories le 500" in result
    assert "sodium le 1000" in result
    assert "author eq 'Jamie Oliver'" in result


def test_recipe_search_tool_build_odata_filter_exclude_ingredients():
    """Test OData filter building with exclude_ingredients."""
    from pantry_agent.tools.recipes import _build_odata_filter

    filters = {"exclude_ingredients": ["tomato", "onion"]}
    result = _build_odata_filter(filters)

    assert "not (search.ismatch('tomato', 'ingredients') or search.ismatch('onion', 'ingredients'))" in result


def test_recipe_search_tool_build_odata_filter_empty():
    """Test OData filter building returns empty string for empty filters."""
    from pantry_agent.tools.recipes import _build_odata_filter

    result = _build_odata_filter({})
    assert result == ""

    result = _build_odata_filter(None)
    assert result == ""


def test_recipe_search_tool_successful_search():
    """Test successful recipe search with mocked Azure Search client."""
    from pantry_agent.tools.recipes import recipe_search_tool

    mock_search_results = [
        {
            "@search.score": 0.92,
            "id": "beef-stew-1",
            "title": "Classic Beef Stew",
            "url": "https://example.com/beef-stew",
            "image": "https://example.com/images/beef-stew.jpg",
            "total_time": 180,
            "calories": 450.0,
            "protein": 35.0,
        },
        {
            "@search.score": 0.85,
            "title": "Quick Beef Chili",
            "url": "https://example.com/beef-chili",
            "image": None,
            "total_time": 45,
            "calories": 380.0,
            "protein": 28.0,
        },
    ]

    with patch("pantry_agent.tools.recipes.settings") as mock_settings, patch(
        "pantry_agent.tools.recipes._generate_embedding"
    ) as mock_embed, patch("pantry_agent.tools.recipes._get_search_client") as mock_search_client_fn:

        mock_settings.azure_search_endpoint = "https://test.search.windows.net"
        mock_settings.azure_search_key = "key"
        mock_settings.azure_search_index = "recipes-index"
        mock_settings.azure_search_semantic_config = "recipe-semantic-config"

        mock_embed.return_value = [0.1] * 1536

        mock_search_client = type("SearchClient", (object,), {
            "search": lambda self, **kwargs: mock_search_results,
        })()
        mock_search_client_fn.return_value = mock_search_client

        result = recipe_search_tool.invoke({"query": "hearty beef stew", "filters": None})

        assert "recipes" in result
        assert len(result["recipes"]) == 2
        assert result["recipes"][0]["id"] == "beef-stew-1"
        assert result["recipes"][0]["title"] == "Classic Beef Stew"
        assert result["recipes"][0]["hybrid_score"] <= 1.0
        assert result["query"] == "hearty beef stew"
        assert "execution_time_ms" in result


def test_recipe_search_tool_search_with_filters():
    """Test recipe search with filters applied."""
    from pantry_agent.tools.recipes import recipe_search_tool

    mock_search_results = [
        {
            "@search.score": 0.88,
            "title": "Healthy Beef Bowl",
            "url": "https://example.com/beef-bowl",
            "image": None,
            "total_time": 30,
            "calories": 350.0,
            "protein": 40.0,
        }
    ]

    with patch("pantry_agent.tools.recipes.settings") as mock_settings, patch(
        "pantry_agent.tools.recipes._generate_embedding"
    ) as mock_embed, patch("pantry_agent.tools.recipes._get_search_client") as mock_search_client_fn, patch(
        "pantry_agent.tools.recipes._build_odata_filter"
    ) as mock_odata:

        mock_settings.azure_search_endpoint = "https://test.search.windows.net"
        mock_settings.azure_search_key = "key"
        mock_settings.azure_search_index = "recipes-index"
        mock_settings.azure_search_semantic_config = "recipe-semantic-config"

        mock_embed.return_value = [0.1] * 1536
        mock_odata.return_value = "total_time le 30 and calories le 500"

        mock_search_client = type("SearchClient", (object,), {
            "search": lambda self, **kwargs: mock_search_results,
        })()
        mock_search_client_fn.return_value = mock_search_client

        filters = {"max_time": 30, "max_calories": 500}
        result = recipe_search_tool.invoke({"query": "quick beef", "filters": filters})

        assert len(result["recipes"]) == 1
        assert result["recipes"][0]["total_time"] == 30
        mock_odata.assert_called_once_with(filters)


def test_recipe_search_tool_handles_exception():
    """Test graceful error handling when search fails."""
    from pantry_agent.tools.recipes import recipe_search_tool

    with patch("pantry_agent.tools.recipes.settings") as mock_settings, patch(
        "pantry_agent.tools.recipes._generate_embedding"
    ) as mock_embed, patch("pantry_agent.tools.recipes._get_search_client") as mock_search_client_fn:

        mock_settings.azure_search_endpoint = "https://test.search.windows.net"
        mock_settings.azure_search_key = "key"
        mock_settings.azure_search_index = "recipes-index"
        mock_settings.azure_search_semantic_config = "recipe-semantic-config"

        mock_embed.return_value = [0.1] * 1536
        mock_search_client_fn.side_effect = Exception("Connection timeout")

        result = recipe_search_tool.invoke({"query": "test"})

        assert result["recipes"] == []
        assert result["total_found"] == 0
        assert "error" in result
        assert "Connection timeout" in result["error"]


def test_recipe_search_tool_output_schema_compliance():
    """Test that output matches RecipeSearchResponse schema."""
    from pantry_agent.tools.recipes import recipe_search_tool
    from pantry_agent.models.schemas import RecipeSearchResponse

    mock_search_results = [
        {
            "@search.score": 0.95,
            "title": "Test Recipe",
            "url": "https://example.com/recipe",
            "image": "https://example.com/image.jpg",
            "total_time": 60,
            "calories": 400.0,
            "protein": 25.0,
        }
    ]

    with patch("pantry_agent.tools.recipes.settings") as mock_settings, patch(
        "pantry_agent.tools.recipes._generate_embedding"
    ) as mock_embed, patch("pantry_agent.tools.recipes._get_search_client") as mock_search_client_fn:

        mock_settings.azure_search_endpoint = "https://test.search.windows.net"
        mock_settings.azure_search_key = "key"
        mock_settings.azure_search_index = "recipes-index"
        mock_settings.azure_search_semantic_config = "recipe-semantic-config"

        mock_embed.return_value = [0.1] * 1536

        mock_search_client = type("SearchClient", (object,), {
            "search": lambda self, **kwargs: mock_search_results,
        })()
        mock_search_client_fn.return_value = mock_search_client

        result = recipe_search_tool.invoke({"query": "test recipe"})

        # Validate it matches the schema by attempting to instantiate
        response = RecipeSearchResponse(**result)
        assert response.query == "test recipe"
        assert len(response.recipes) == 1
        assert response.recipes[0].title == "Test Recipe"


def test_recipe_search_tool_falls_back_when_embedding_fails():
    """Embedding 404s should not prevent semantic text search results."""
    from pantry_agent.tools.recipes import recipe_search_tool

    mock_search_results = [
        {
            "@search.score": 0.75,
            "title": "Quick Vegan Pasta",
            "url": "https://example.com/vegan-pasta",
            "image": None,
            "total_time": 20,
            "calories": 320.0,
            "protein": 12.0,
        }
    ]

    with patch("pantry_agent.tools.recipes.settings") as mock_settings, patch(
        "pantry_agent.tools.recipes._generate_embedding"
    ) as mock_embed, patch("pantry_agent.tools.recipes._get_search_client") as mock_search_client_fn:
        mock_settings.azure_search_endpoint = "https://test.search.windows.net"
        mock_settings.azure_search_key = "key"
        mock_settings.azure_search_index = "recipes-index"
        mock_settings.azure_search_semantic_config = "recipe-semantic-config"

        mock_embed.side_effect = Exception("Resource not found")

        mock_search_client = MagicMock()
        mock_search_client.search.return_value = mock_search_results
        mock_search_client_fn.return_value = mock_search_client

        result = recipe_search_tool.invoke({"query": "quick vegan dinner"})

        assert len(result["recipes"]) == 1
        assert result["recipes"][0]["title"] == "Quick Vegan Pasta"
        search_kwargs = mock_search_client.search.call_args.kwargs
        assert "vectors" not in search_kwargs


def test_get_recipe_details_uses_pantry_analysis(sample_pantry_items):
    """Recipe details should compare ingredients against pantry items and surface substitutions."""
    from pantry_agent.tools.recipes import get_recipe_details

    mock_recipe = {
        "id": "r2",
        "name": "Creamy Spinach Pasta",
        "description": "A simple pasta with a creamy sauce.",
        "ingredients": [
            {"name": "eggs", "quantity": 3, "unit": "pieces"},
            {"name": "spinach", "quantity": 100, "unit": "g"},
            {"name": "parmesan", "quantity": 1, "unit": "cup"},
        ],
        "instructions": ["Boil pasta", "Mix sauce", "Serve warm"],
        "url": "https://example.com/creamy-spinach-pasta",
    }
    pantry_with_substitute = sample_pantry_items + [
        {
            "id": "5",
            "name": "nutritional yeast",
            "quantity": 1.0,
            "unit": "cup",
            "category": "condiments",
        }
    ]

    with patch("pantry_agent.tools.pantry.safe_api_call", return_value={"items": pantry_with_substitute, "total": len(pantry_with_substitute)}):
        result = get_recipe_details.invoke({"recipe": mock_recipe})

    assert result["recipe"]["name"] == "Creamy Spinach Pasta"
    assert "parmesan" in result["missing_ingredients"]
    assert result["substitutions"]["parmesan"] == "nutritional yeast"
    assert result["pantry_coverage_pct"] == 66.7
    assert "eggs" in result["available_ingredients"]
    assert "spinach" in result["available_ingredients"]


def test_get_recipe_details_output_schema_compliance(sample_pantry_items):
    """Recipe details output should satisfy the RecipeDetailResponse schema."""
    from pantry_agent.models.schemas import RecipeDetailResponse
    from pantry_agent.tools.recipes import get_recipe_details

    recipe = {
        "id": "r1",
        "name": "Spinach Omelette",
        "description": "Quick and healthy breakfast",
        "ingredients": [{"name": "eggs", "quantity": 3, "unit": "pieces"}, {"name": "spinach", "quantity": 50, "unit": "g"}],
        "instructions": ["Beat eggs", "Add spinach", "Cook in pan"],
    }

    with patch("pantry_agent.tools.pantry.safe_api_call", return_value={"items": sample_pantry_items, "total": len(sample_pantry_items)}):
        result = get_recipe_details.invoke({"recipe": recipe})

    response = RecipeDetailResponse(**result)
    assert response.recipe.name == "Spinach Omelette"
    assert response.pantry_coverage_pct == 100.0
    assert response.missing_ingredients == []


def test_get_recipe_details_accepts_plain_recipe_name_string():
    """Plain text recipe names should resolve through the details fetch path."""
    from pantry_agent.tools.recipes import get_recipe_details

    fetched = {
        "id": "r3",
        "name": "Tomato Soup",
        "description": "Simple tomato soup.",
        "ingredients": [{"name": "tomato", "quantity": 4, "unit": "pieces"}],
        "instructions": ["Simmer and blend."],
    }

    with patch("pantry_agent.tools.recipes.safe_api_call", return_value={"recipe": fetched}), patch(
        "pantry_agent.tools.pantry.safe_api_call", return_value={"items": [], "total": 0}
    ):
        result = get_recipe_details.invoke({"recipe": "Tomato Soup"})

    assert result["recipe"]["name"] == "Tomato Soup"
    assert result["missing_ingredients"] == ["tomato"]


def test_get_recipe_details_accepts_alias_keys_name_id_url():
    """Alias keys should map to recipe_name/recipe_id/recipe_url resolution."""
    from pantry_agent.tools.recipes import get_recipe_details

    fetched = {
        "id": "r4",
        "name": "Garlic Bread",
        "description": "Crispy garlic bread.",
        "ingredients": [{"name": "bread", "quantity": 1, "unit": "loaf"}],
        "instructions": ["Bake until golden."],
        "url": "https://example.com/garlic-bread",
    }

    with patch("pantry_agent.tools.recipes.safe_api_call", return_value={"recipe": fetched}), patch(
        "pantry_agent.tools.pantry.safe_api_call", return_value={"items": [], "total": 0}
    ):
        result = get_recipe_details.invoke({"name": "Garlic Bread", "id": "r4", "url": "https://example.com/garlic-bread"})

    assert result["recipe"]["id"] == "r4"
    assert result["recipe"]["name"] == "Garlic Bread"


def test_get_recipe_details_uses_selected_recipe_summary_without_backend_fetch():
    """Search-result selections should return a summary without hitting unsupported backend detail routes."""
    from pantry_agent.tools.recipes import get_recipe_details

    selected_recipe = {
        "id": "r4",
        "title": "Garlic Bread",
        "url": "https://example.com/garlic-bread",
    }

    with patch("pantry_agent.tools.recipes.safe_api_call") as mock_safe, patch(
        "pantry_agent.tools.pantry.safe_api_call", return_value={"items": [], "total": 0}
    ):
        result = get_recipe_details.invoke({"recipe": selected_recipe})

    assert result["recipe"]["id"] == "r4"
    assert result["recipe"]["name"] == "Garlic Bread"
    assert result["missing_ingredients"] == []
    assert result["pantry_coverage_pct"] is None
    assert "Full ingredient details are unavailable" in result["message"]
    mock_safe.assert_not_called()


def test_get_recipe_details_uses_collection_endpoint_for_id_lookup():
    """ID-based lookups should use the supported collection endpoint instead of a dead detail route."""
    from pantry_agent.tools.recipes import get_recipe_details

    fetched = {
        "id": "r4",
        "name": "Garlic Bread",
        "description": "Crispy garlic bread.",
        "ingredients": [{"name": "bread", "quantity": 1, "unit": "loaf"}],
        "instructions": ["Bake until golden."],
        "url": "https://example.com/garlic-bread",
    }

    with patch("pantry_agent.tools.recipes.safe_api_call", return_value={"recipe": fetched}) as mock_safe, patch(
        "pantry_agent.tools.pantry.safe_api_call", return_value={"items": [], "total": 0}
    ):
        result = get_recipe_details.invoke({"id": "r4"})

    assert result["recipe"]["id"] == "r4"
    first_call_args = mock_safe.call_args_list[0].args
    assert first_call_args[1] == "/api/ai/recipes"
    assert first_call_args[2] == {"id": "r4"}


def test_get_recipe_details_returns_summary_when_only_search_hit_is_available():
    """Summary-only search results should degrade gracefully without a validation error."""
    from pantry_agent.tools.recipes import get_recipe_details

    selected_recipe = {
        "title": "La Madeleine Tomato Basil Soup",
        "url": "https://example.com/tomato-basil-soup",
        "image": "https://example.com/soup.jpg",
        "total_time": 40,
    }

    with patch("pantry_agent.tools.pantry.safe_api_call", return_value={"items": [], "total": 0}):
        result = get_recipe_details.invoke({"recipe": selected_recipe})

    assert result["recipe"]["name"] == "La Madeleine Tomato Basil Soup"
    assert result["missing_ingredients"] == []
    assert result["available_ingredients"] == []
    assert result["pantry_coverage_pct"] is None
    assert "Full ingredient details are unavailable" in result["message"]


def test_cooking_copilot_sends_recipe_name_field():
    """API payload should always include non-blank recipeName."""
    from pantry_agent.tools.cooking import cooking_copilot

    with patch("pantry_agent.tools.cooking.safe_api_call", return_value={"answer": "ok"}) as mock_safe:
        result = cooking_copilot.invoke(
            {
                "question": "How long should I cook this?",
                "recipe_name": "Tomato Lentil Soup",
            }
        )

    assert result["answer"] == "ok"
    _, _, payload = mock_safe.call_args.args
    assert payload["recipeName"] == "Tomato Lentil Soup"


def test_cooking_copilot_defaults_recipe_name_when_missing():
    """When no recipe is provided, payload still satisfies backend validation."""
    from pantry_agent.tools.cooking import cooking_copilot

    with patch("pantry_agent.tools.cooking.safe_api_call", return_value={"answer": "ok"}) as mock_safe:
        _ = cooking_copilot.invoke({"question": "Need a quick tip"})

    _, _, payload = mock_safe.call_args.args
    assert payload["recipeName"] == "general"
