"""
Recipe Intelligence Tool
========================
Recommend recipes that can be cooked using the user's current pantry contents.
"""
from __future__ import annotations

import logging
import os
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

    # Exclude ingredients (search.in function for array field)
    if exclude_ingredients := filters.get("exclude_ingredients"):
        if isinstance(exclude_ingredients, list):
            excluded = ", ".join(f"'{ing}'" for ing in exclude_ingredients)
            conditions.append(f"search.in(ingredients, {excluded})")

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
            "error": f"Recipe search failed: {str(e)}",
        }
