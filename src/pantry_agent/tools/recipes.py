"""
Recipe Intelligence Tool
========================
Recommend recipes that can be cooked using the user's current pantry contents.
"""
from __future__ import annotations

import logging
import os
import re
import time
from functools import lru_cache
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from langchain_core.tools import tool
from openai import AzureOpenAI
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from ..api_client import api_get, safe_api_call
from ..config import settings
from ..models.schemas import RecipeSearchInput, RecipeSearchResponse, RecipeSearchResult

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


class RecipeDetailsInput(BaseModel):
    recipe_id: str | None = Field(default=None, description="Stable recipe identifier, if available.")
    recipe_name: str | None = Field(default=None, description="Recipe name to resolve when no ID is available.")
    recipe_url: str | None = Field(default=None, description="Canonical recipe URL when selecting from search results.")
    id: str | None = Field(
        default=None,
        description="Alias for recipe_id for callers that send generic 'id'.",
    )
    name: str | None = Field(
        default=None,
        description="Alias for recipe_name so plain recipe name inputs are accepted.",
    )
    url: str | None = Field(
        default=None,
        description="Alias for recipe_url for callers that send generic 'url'.",
    )
    recipe: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Optional recipe selection payload from a prior recommendation/search result, "
            "or a plain recipe name string."
        ),
    )
    include_pantry_analysis: bool = Field(
        default=True,
        description="Compare recipe ingredients against the current pantry and return missing items plus substitutions.",
    )


_DEFAULT_SUBSTITUTIONS: dict[str, list[str]] = {
    "butter": ["olive oil", "ghee", "coconut oil"],
    "milk": ["oat milk", "soy milk", "water"],
    "heavy cream": ["coconut cream", "evaporated milk"],
    "eggs": ["flaxseed meal", "chia seed gel", "applesauce"],
    "flour": ["oat flour", "almond flour", "cornstarch"],
    "sugar": ["honey", "maple syrup", "date syrup"],
    "spinach": ["kale", "swiss chard", "mixed greens"],
    "tomato": ["tomato paste", "canned tomatoes", "red pepper"],
    "chicken breast": ["tofu", "chickpeas", "turkey breast"],
    "parmesan": ["pecorino romano", "nutritional yeast"],
    "yogurt": ["sour cream", "coconut yogurt", "Greek yogurt"],
}


def _normalize_ingredient_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    words: list[str] = []
    for word in normalized.split():
        if len(word) > 3 and word.endswith("ies"):
            words.append(word[:-3] + "y")
        elif len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
            words.append(word[:-2])
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            words.append(word[:-1])
        else:
            words.append(word)
    return " ".join(words)


def _ingredient_label(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("name", "ingredient", "title", "label"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _recipe_title(recipe: dict[str, Any]) -> str:
    return str(recipe.get("name") or recipe.get("title") or "").strip()


def _recipe_identifier(recipe: dict[str, Any]) -> str | None:
    value = recipe.get("id") or recipe.get("recipe_id")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _recipe_url(recipe: dict[str, Any]) -> str | None:
    value = recipe.get("url") or recipe.get("recipe_url") or recipe.get("source_url")
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _recipe_has_detail_fields(recipe: dict[str, Any]) -> bool:
    return bool(recipe.get("ingredients") or recipe.get("instructions") or recipe.get("description"))


def _recipe_has_summary_fields(recipe: dict[str, Any]) -> bool:
    return bool(_recipe_title(recipe) or _recipe_url(recipe) or _recipe_identifier(recipe))


def _normalize_recipe_summary(recipe: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(recipe)
    title = _recipe_title(normalized)
    if title and not normalized.get("name"):
        normalized["name"] = title
    if "title" in normalized and normalized.get("name") == normalized.get("title"):
        normalized.pop("title", None)
    return normalized


def _match_recipe_candidate(candidate: dict[str, Any], recipe_id: str | None, recipe_name: str | None, recipe_url: str | None) -> bool:
    if recipe_id and str(candidate.get("id")) == str(recipe_id):
        return True
    if recipe_url and candidate.get("url") == recipe_url:
        return True
    if recipe_name and _normalize_ingredient_name(_recipe_title(candidate)) == _normalize_ingredient_name(recipe_name):
        return True
    return False


def _coerce_recipe_payload(payload: Any, recipe_id: str | None, recipe_name: str | None, recipe_url: str | None) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if isinstance(payload.get("recipe"), dict):
            candidate = payload["recipe"]
            if _match_recipe_candidate(candidate, recipe_id, recipe_name, recipe_url):
                return candidate
        if isinstance(payload.get("recipes"), list):
            payload = payload["recipes"]
        elif _recipe_has_detail_fields(payload):
            return payload
    if isinstance(payload, list):
        for candidate in payload:
            if isinstance(candidate, dict) and _match_recipe_candidate(candidate, recipe_id, recipe_name, recipe_url):
                return candidate
        for candidate in payload:
            if isinstance(candidate, dict) and _recipe_has_detail_fields(candidate):
                return candidate
    return None


def _fetch_recipe_details(recipe_id: str | None, recipe_name: str | None, recipe_url: str | None) -> dict[str, Any] | None:
    candidates: list[tuple[str, dict[str, Any] | None]] = []
    if recipe_id:
        candidates.extend(
            [
                ("/api/ai/recipes", {"id": recipe_id}),
            ]
        )
    if recipe_name:
        candidates.extend(
            [
                ("/api/ai/recipes", {"name": recipe_name}),
            ]
        )
    if recipe_url:
        candidates.extend(
            [
                ("/api/ai/recipes", {"url": recipe_url}),
            ]
        )

    for path, params in candidates:
        result = safe_api_call(api_get, path, params or None)
        if isinstance(result, dict) and result.get("error"):
            continue
        recipe = _coerce_recipe_payload(result, recipe_id, recipe_name, recipe_url)
        if recipe:
            return recipe
    return None


def _extract_recipe_ingredients(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    ingredients = recipe.get("ingredients", [])
    if not isinstance(ingredients, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in ingredients:
        label = _ingredient_label(item)
        if not label:
            continue
        quantity = item.get("quantity") if isinstance(item, dict) else None
        unit = item.get("unit") if isinstance(item, dict) else None
        normalized.append({"name": label, "quantity": quantity, "unit": unit, "raw": item})
    return normalized


def _suggest_substitution(ingredient_name: str, pantry_lookup: dict[str, dict[str, Any]]) -> str | None:
    normalized = _normalize_ingredient_name(ingredient_name)
    options = _DEFAULT_SUBSTITUTIONS.get(normalized, [])
    for option in options:
        if _normalize_ingredient_name(option) in pantry_lookup:
            return option
    return options[0] if options else None


def _build_recommendation_query(cuisine: str | None, meal_type: str | None) -> str:
    """
    Build a natural language query from recommendation parameters.
    Falls back to a generic recommendation query if no specific filters are provided.
    """
    parts = []
    if cuisine:
        parts.append(f"{cuisine} style")
    if meal_type:
        parts.append(meal_type)
    if parts:
        query = " ".join(parts) + " recipe idea"
    else:
        query = "recipe recommendation"
    return query


def _rank_by_pantry_coverage(
    recipes: list[dict[str, Any]],
    pantry_items: list[dict[str, Any]],
    max_missing_ingredients: int | None = None,
) -> list[dict[str, Any]]:
    """
    Post-filter recipes by max_missing_ingredients and rank by pantry coverage.
    Returns recipes sorted by pantry coverage percentage (highest first).
    """
    # Build pantry lookup
    pantry_lookup = {
        _normalize_ingredient_name(item.get("name", "")): item
        for item in pantry_items
        if isinstance(item, dict) and _normalize_ingredient_name(item.get("name", ""))
    }

    ranked_recipes = []
    for recipe in recipes:
        # Skip recipes that don't have ingredient info
        ingredients = recipe.get("ingredients", [])
        if not isinstance(ingredients, list):
            continue

        if not ingredients:
            # No ingredient details available, include but with 0% coverage
            recipe["pantry_coverage_pct"] = 0.0
            recipe["missing_ingredients"] = []
            recipe["available_ingredients"] = []
            ranked_recipes.append(recipe)
            continue

        # Compare ingredients against pantry
        available_count = 0
        missing_count = 0
        missing_list = []
        available_list = []

        for ingredient in ingredients:
            ingredient_name = _ingredient_label(ingredient) if isinstance(ingredient, dict) else str(ingredient)
            if not ingredient_name:
                continue

            normalized = _normalize_ingredient_name(ingredient_name)
            is_available = normalized in pantry_lookup or any(
                normalized in pantry_key or pantry_key in normalized for pantry_key in pantry_lookup
            )

            if is_available:
                available_count += 1
                available_list.append(ingredient_name)
            else:
                missing_count += 1
                missing_list.append(ingredient_name)

        total_ingredients = len(ingredients)
        coverage_pct = round((available_count / total_ingredients) * 100, 1) if total_ingredients else 0.0

        # Filter by max_missing_ingredients if specified
        if max_missing_ingredients is not None and missing_count > max_missing_ingredients:
            continue

        recipe["pantry_coverage_pct"] = coverage_pct
        recipe["missing_ingredients"] = missing_list
        recipe["available_ingredients"] = available_list
        ranked_recipes.append(recipe)

    # Sort by pantry coverage (highest first)
    ranked_recipes.sort(key=lambda r: r.get("pantry_coverage_pct", 0.0), reverse=True)

    return ranked_recipes


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
    Searches for recipe recommendations personalised to the user's available
    ingredients using Azure Search hybrid search. Recipes are ranked by pantry
    coverage (highest coverage first).

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
    - For detailed cooking instructions or a selected recipe → use get_recipe_details
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
          "tags": ["pasta", "quick"],
          "pantry_coverage_pct": 85.0,
          "available_ingredients": [...],
          "missing_ingredients": [...]
        }
      ],
      "message": "Found 8 recipes you can cook now."
    }

    DEPENDENCIES
    ------------
    Requires Azure Search service configured. For best results, call
    get_pantry_inventory first so the pantry lookup is accurate.
    """
    try:
        # Build natural language query from filters
        query = _build_recommendation_query(cuisine, meal_type)

        # Build search filters
        search_filters: dict[str, Any] = {}
        if max_cook_time_minutes is not None:
            search_filters["max_time"] = max_cook_time_minutes
        if cuisine:
            search_filters["cuisine"] = cuisine
        if meal_type:
            search_filters["meal_type"] = meal_type

        # Execute search
        search_results = recipe_search_tool.invoke({"query": query, "filters": search_filters or None})

        if search_results.get("error"):
            search_error = search_results.get("message") or search_results.get("error") or "unknown error"
            return {
                "recipes": [],
                "message": f"Recipe recommendations unavailable: {search_error}",
            }

        # Fetch pantry inventory for post-filtering and coverage analysis
        from .pantry import get_pantry_inventory

        pantry_result = get_pantry_inventory.invoke({})
        pantry_items = pantry_result.get("items", []) if isinstance(pantry_result, dict) else []

        # Get recipes from search results and rank by pantry coverage
        recipes_raw = search_results.get("recipes", [])
        ranked_recipes = _rank_by_pantry_coverage(recipes_raw, pantry_items, max_missing_ingredients)

        # Generate summary message
        if ranked_recipes:
            cookable_now = sum(1 for r in ranked_recipes if r.get("pantry_coverage_pct", 0.0) == 100.0)
            message = f"Found {len(ranked_recipes)} recipe recommendations"
            if cookable_now > 0:
                message += f" ({cookable_now} you can cook right now)"
            message += "."
        else:
            message = "No recipes found matching your criteria and pantry."

        result = {
            "recipes": ranked_recipes,
            "message": message,
        }

        logger.info(
            "Recipe recommendations: query=%r, cuisine=%r, meal_type=%r, found=%d",
            query,
            cuisine,
            meal_type,
            len(ranked_recipes),
        )
        return result

    except Exception as e:
        logger.warning("Recipe recommendations failed: %s", e)
        return {
            "recipes": [],
            "message": f"Recipe recommendations failed: {str(e)}",
            "error": str(e),
        }


@tool(args_schema=RecipeDetailsInput)
def get_recipe_details(
    recipe_id: str | None = None,
    recipe_name: str | None = None,
    recipe_url: str | None = None,
    id: str | None = None,
    name: str | None = None,
    url: str | None = None,
    recipe: dict[str, Any] | str | None = None,
    include_pantry_analysis: bool = True,
) -> dict[str, Any]:
    """
    Return a selected recipe plus pantry-aware missing ingredient analysis.

    PURPOSE
    -------
    Fetches or reuses a recipe payload and compares its ingredients against the
    user's current pantry so the caller can see what is already available,
    what is missing, and which substitutions are reasonable.
    """
    resolved_recipe_id = recipe_id or id
    resolved_recipe_name = recipe_name or name
    resolved_recipe_url = recipe_url or url

    # Accept direct recipe-name inputs when callers pass recipe as plain text.
    if isinstance(recipe, str):
        if recipe.strip() and not resolved_recipe_name:
            resolved_recipe_name = recipe.strip()
        selected_recipe: dict[str, Any] = {}
    else:
        selected_recipe = dict(recipe or {})
        if selected_recipe:
            selected_recipe = _normalize_recipe_summary(selected_recipe)

    if not resolved_recipe_id:
        resolved_recipe_id = _recipe_identifier(selected_recipe)
    if not resolved_recipe_url:
        resolved_recipe_url = _recipe_url(selected_recipe)

    if not resolved_recipe_name:
        selected_title = _recipe_title(selected_recipe)
        if selected_title:
            resolved_recipe_name = selected_title

    should_fetch_recipe_details = not selected_recipe or not _recipe_has_summary_fields(selected_recipe)
    if not _recipe_has_detail_fields(selected_recipe) and should_fetch_recipe_details:
        fetched_recipe = _fetch_recipe_details(
            resolved_recipe_id,
            resolved_recipe_name,
            resolved_recipe_url,
        )
        if fetched_recipe:
            selected_recipe = {**selected_recipe, **fetched_recipe}

    if not selected_recipe:
        return {
            "error": True,
            "message": "Recipe details are unavailable for the selected recipe.",
        }

    selected_recipe = _normalize_recipe_summary(selected_recipe)

    recipe_title = _recipe_title(selected_recipe)
    response: dict[str, Any] = {"recipe": selected_recipe}

    if not include_pantry_analysis:
        response["message"] = f"Loaded details for {recipe_title or 'selected recipe'}."
        return response

    from .pantry import get_pantry_inventory

    pantry_result = get_pantry_inventory.invoke({})
    pantry_items = pantry_result.get("items", []) if isinstance(pantry_result, dict) else []
    pantry_lookup = {
        _normalize_ingredient_name(item.get("name", "")): item
        for item in pantry_items
        if isinstance(item, dict) and _normalize_ingredient_name(item.get("name", ""))
    }

    ingredient_rows = _extract_recipe_ingredients(selected_recipe)
    available_ingredients: list[str] = []
    missing_ingredients: list[str] = []
    pantry_matches: list[str] = []
    substitutions: dict[str, str] = {}
    shopping_list: list[dict[str, Any]] = []

    for ingredient in ingredient_rows:
        ingredient_name = ingredient["name"]
        normalized = _normalize_ingredient_name(ingredient_name)
        is_available = normalized in pantry_lookup or any(
            normalized in pantry_key or pantry_key in normalized for pantry_key in pantry_lookup
        )

        if is_available:
            available_ingredients.append(ingredient_name)
            pantry_matches.append(ingredient_name)
            continue

        missing_ingredients.append(ingredient_name)
        shopping_list.append(
            {
                "name": ingredient_name,
                "quantity": ingredient.get("quantity"),
                "unit": ingredient.get("unit"),
            }
        )
        if substitution := _suggest_substitution(ingredient_name, pantry_lookup):
            substitutions[ingredient_name] = substitution

    total_ingredients = len(ingredient_rows)
    pantry_coverage_pct = round((len(available_ingredients) / total_ingredients) * 100, 1) if total_ingredients else None

    if ingredient_rows:
        message = (
            f"Recipe details prepared for {recipe_title or 'selected recipe'}. "
            f"{len(available_ingredients)} ingredient(s) on hand, {len(missing_ingredients)} missing."
        )
    else:
        message = (
            f"Loaded recipe summary for {recipe_title or 'selected recipe'}. "
            "Full ingredient details are unavailable from the current source."
        )

    response.update(
        {
            "pantry_coverage_pct": pantry_coverage_pct,
            "available_ingredients": available_ingredients,
            "missing_ingredients": missing_ingredients,
            "substitutions": substitutions,
            "pantry_matches": pantry_matches,
            "shopping_list": shopping_list,
            "message": message,
        }
    )

    logger.info(
        "Recipe details resolved: recipe=%r, pantry_coverage=%s%%, missing=%d",
        recipe_title,
        pantry_coverage_pct,
        len(missing_ingredients),
    )
    return response


# ── Azure Search Hybrid Recipe Search ────────────────────────────────────────


def _build_odata_filter(filters: dict) -> str:
    """
    Convert filter dict to OData filter string for Azure Search.

    Example:
        {"max_time": 30, "min_protein": 10} → "total_time le 30 and protein ge 10"
    """
    if not filters:
        return ""

    conditions = []

    # Numeric comparisons
    if max_time := filters.get("max_time"):
        conditions.append(f"total_time le {max_time}")
    if min_protein := filters.get("min_protein"):
        conditions.append(f"protein ge {min_protein}")
    if max_calories := filters.get("max_calories"):
        conditions.append(f"calories le {max_calories}")
    if max_sodium := filters.get("max_sodium"):
        conditions.append(f"sodium le {max_sodium}")
    if max_fat := filters.get("max_fat"):
        conditions.append(f"fat le {max_fat}")
    if max_carbohydrate := filters.get("max_carbohydrate"):
        conditions.append(f"carbohydrate le {max_carbohydrate}")
    if max_cholesterol := filters.get("max_cholesterol"):
        conditions.append(f"cholesterol le {max_cholesterol}")
    if min_fiber := filters.get("min_fiber"):
        conditions.append(f"fiber ge {min_fiber}")
    if max_sugar := filters.get("max_sugar"):
        conditions.append(f"sugar le {max_sugar}")

    # String filters
    if author := filters.get("author"):
        conditions.append(f"author eq '{author}'")
    if yields := filters.get("yields"):
        conditions.append(f"yields eq '{yields}'")
    if cuisine := filters.get("cuisine"):
        conditions.append(f"cuisine eq '{cuisine}'")
    if meal_type := filters.get("meal_type"):
        conditions.append(f"meal_type eq '{meal_type}'")

    # Exclude ingredients using fielded full-text match.
    # This avoids any()/all() syntax so the filter works whether ingredients
    # is indexed as a string or as a collection.
    if exclude_ingredients := filters.get("exclude_ingredients"):
        if isinstance(exclude_ingredients, list):
            terms = [str(ing).strip() for ing in exclude_ingredients if str(ing).strip()]
            if terms:
                escaped_terms = [term.replace("'", "''") for term in terms]
                ingredient_matches = [
                    f"search.ismatch('" + term + "', 'ingredients')" for term in escaped_terms
                ]
                conditions.append(f"not ({' or '.join(ingredient_matches)})")

    return " and ".join(conditions)


@lru_cache(maxsize=1)
def _get_embedding_client() -> AzureOpenAI:
    """Singleton Azure OpenAI client for embeddings."""
    # azure_endpoint must be the resource base URL (scheme + host only).
    # settings.azure_openai_endpoint may include an OpenAI-compat path
    # like /openai/v1 which is only valid for base_url; strip it here.
    parsed = urlparse(settings.azure_openai_endpoint)
    resource_endpoint = f"{parsed.scheme}://{parsed.netloc}/"
    return AzureOpenAI(
        api_key=settings.openai_api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=resource_endpoint,
    )


def _generate_embedding(query: str) -> list[float]:
    """Generate embedding vector for the query using Azure OpenAI."""
    try:
        client = _get_embedding_client()
        deployment_name = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", settings.azure_openai_embedding_deployment
        )
        response = client.embeddings.create(
            input=query,
            model=deployment_name,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning("Failed to generate embedding for query: %s", e)
        raise


@lru_cache(maxsize=1)
def _get_search_client() -> SearchClient:
    """Singleton Azure Search client."""
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index,
        credential=AzureKeyCredential(settings.azure_search_key),
    )


@tool(args_schema=RecipeSearchInput)
def recipe_search_tool(query: str, filters: dict | None = None) -> dict[str, Any]:
    """
    Search for recipes using hybrid search (BM25 + vector embeddings + semantic ranking)
    against Azure AI Search index built from Food.com recipe data.

    PURPOSE
    -------
    Performs a deep semantic search for recipes, combining keyword matching (BM25),
    dense vector embedding similarity, and semantic ranking to find the most relevant
    recipes based on natural language queries.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "Find me a recipe for..." with specific flavors/ingredients
    - Asks "What's a good recipe for [specific cuisine/dietary need]?"
    - Asks for recipe search with detailed nutritional or time constraints
    - Wants highly relevant recipe discovery beyond pantry-based matching

    WHEN NOT TO USE
    ---------------
    - For pantry-based recommendations → use recommend_recipes
    - For meal planning across multiple days → use create_diet_plan
    - For cooking instructions → use cooking_copilot

    INPUTS
    ------
    query: str
        Natural language recipe search (e.g., "hearty beef stew for winter", "quick vegan dinner")
    filters: dict | None
        Optional filter dict with keys:
        - max_time (int): max cook time in minutes
        - min_protein (float): minimum protein in grams
        - max_calories (float): maximum calories per serving
        - max_sodium (float): maximum sodium in mg
        - max_fat, max_carbohydrate, max_cholesterol, min_fiber, max_sugar: numeric filters
        - author (str): specific recipe author
        - yields (str): serving size
        - exclude_ingredients (list): ingredients to exclude

    RETURNS
    -------
    {
      "recipes": [
        {
          "title": "Classic Beef Stew",
          "url": "https://...",
          "image": "https://...",
          "total_time": 180,
          "calories": 450.0,
          "protein": 35.0,
          "hybrid_score": 0.92
        },
        ...
      ],
      "total_found": 1240,
      "query": "hearty beef stew for winter",
      "execution_time_ms": 245.3
    }

    DEPENDENCIES
    ------------
    Requires Azure Search service with:
    - Index name: recipes-index (configurable via AZURE_SEARCH_INDEX)
    - Semantic configuration: recipe-semantic-config (configurable via settings)
    - Vector field: content_vector (1536 dimensions, cosine similarity)
    - Embedding model: text-embedding-ada-002 (configurable via AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    """
    start_time = time.time()

    try:
        # Validate Azure Search config
        if not settings.azure_search_endpoint or not settings.azure_search_key:
            return {
                "recipes": [],
                "total_found": 0,
                "query": query,
                "execution_time_ms": None,
                "message": "Azure Search not configured (missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_KEY)",
                "error": "Azure Search not configured (missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_KEY)",
            }

        # Generate embedding for the query. If embeddings are unavailable
        # (e.g. missing deployment), degrade to semantic text-only search.
        embedding: list[float] | None = None
        try:
            embedding = _generate_embedding(query)
        except Exception as exc:
            logger.warning("Embedding unavailable, falling back to semantic text search: %s", exc)

        # Build OData filter string
        odata_filter = _build_odata_filter(filters or {})

        # Get search client
        search_client = _get_search_client()

        search_kwargs: dict[str, Any] = {
            "search_text": query,
            "query_type": QueryType.SEMANTIC,
            "semantic_configuration_name": settings.azure_search_semantic_config,
            "filter": odata_filter if odata_filter else None,
            "top": 5,
            "select": [
                "id",
                "title",
                "url",
                "image",
                "total_time",
                "calories",
                "protein",
            ],
        }
        if embedding is not None:
            search_kwargs["vector_queries"] = [
                VectorizedQuery(
                    vector=embedding,
                    k_nearest_neighbors=5,
                    fields="content_vector",
                )
            ]

        # Execute hybrid search when embeddings are available, semantic search otherwise.
        results = search_client.search(**search_kwargs)

        # Extract and format results
        recipes = []
        total_found = 0

        for result in results:
            # Normalize score to 0-1 range (Azure returns scores that can exceed 1)
            hybrid_score = min(result["@search.score"] / 100.0, 1.0) if result.get("@search.score") else 0.0

            recipe = RecipeSearchResult(
                id=result.get("id"),
                title=result.get("title", "Unknown"),
                url=result.get("url", ""),
                image=result.get("image"),
                total_time=result.get("total_time"),
                calories=result.get("calories"),
                protein=result.get("protein"),
                hybrid_score=hybrid_score,
            )
            recipes.append(recipe)

            # Capture total count from search result metadata
            if not total_found and hasattr(results, "get_count"):
                total_found = results.get_count()

        execution_time_ms = (time.time() - start_time) * 1000

        response_dict = {
            "recipes": [r.model_dump() for r in recipes],
            "total_found": total_found or len(recipes),
            "query": query,
            "execution_time_ms": execution_time_ms,
        }

        logger.info(
            "Recipe search: query=%r, filters=%r, found=%d, time=%.1fms",
            query,
            filters,
            len(recipes),
            execution_time_ms,
        )

        return response_dict

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        logger.warning(
            "Recipe search failed: query=%r, error=%s, time=%.1fms",
            query,
            e,
            execution_time_ms,
        )
        # Graceful fallback: return empty list with error message
        return {
            "recipes": [],
            "total_found": 0,
            "query": query,
            "execution_time_ms": execution_time_ms,
            "message": f"Recipe search failed: {str(e)}",
            "error": f"Recipe search failed: {str(e)}",
        }
