"""
Pantry Management Tools
=======================
CRUD operations on the user's personal pantry inventory.
Never hallucinate pantry items – always reflect real API responses.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_get, api_post, safe_api_call

logger = logging.getLogger(__name__)

_DEFAULT_USER_ID = "default_user"
_CURRENT_USER_ID: ContextVar[str] = ContextVar("pantry_tool_user_id", default=_DEFAULT_USER_ID)
_BACKEND_CATEGORIES = {
    "OTHER",
    "VEGETABLE",
    "DAIRY",
    "FRUIT",
    "MEAT",
    "GROCERY",
    "SNACK",
    "BEVERAGE",
    "BAKERY",
}
_CATEGORY_ALIASES = {
    "other": "OTHER",
    "produce": "VEGETABLE",
    "vegetable": "VEGETABLE",
    "vegetables": "VEGETABLE",
    "veg": "VEGETABLE",
    "fruit": "FRUIT",
    "fruits": "FRUIT",
    "dairy": "DAIRY",
    "meat": "MEAT",
    "protein": "MEAT",
    "seafood": "MEAT",
    "fish": "MEAT",
    "grocery": "GROCERY",
    "groceries": "GROCERY",
    "frozen": "GROCERY",
    "canned": "GROCERY",
    "dry_goods": "GROCERY",
    "dry goods": "GROCERY",
    "condiments": "GROCERY",
    "condiment": "GROCERY",
    "pantry": "GROCERY",
    "snack": "SNACK",
    "snacks": "SNACK",
    "beverage": "BEVERAGE",
    "beverages": "BEVERAGE",
    "drink": "BEVERAGE",
    "drinks": "BEVERAGE",
    "bakery": "BAKERY",
    "bread": "BAKERY",
}
_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DAIRY", ("milk", "cheese", "yogurt", "yoghurt", "butter", "cream", "curd", "paneer", "egg")),
    ("BEVERAGE", ("cola", "juice", "soda", "water", "coffee", "tea", "drink", "beverage")),
    ("BAKERY", ("bread", "bun", "bagel", "croissant", "cake", "muffin", "biscuit", "pastry", "loaf")),
    ("SNACK", ("chips", "crisps", "cracker", "cookie", "cookies", "chocolate", "candy", "popcorn", "bar")),
    ("FRUIT", ("apple", "banana", "orange", "grape", "berry", "berries", "mango", "pineapple", "melon", "pear")),
    ("VEGETABLE", ("tomato", "onion", "potato", "carrot", "spinach", "lettuce", "broccoli", "pepper", "cucumber")),
    ("MEAT", ("chicken", "beef", "pork", "lamb", "turkey", "fish", "salmon", "tuna", "shrimp", "sausage")),
    ("GROCERY", ("rice", "pasta", "flour", "sugar", "salt", "oil", "sauce", "beans", "lentils", "spice")),
)


def set_pantry_tool_user_id(user_id: str | None) -> Token[str]:
    normalized_user_id = (user_id or "").strip() or _DEFAULT_USER_ID
    return _CURRENT_USER_ID.set(normalized_user_id)


def reset_pantry_tool_user_id(token: Token[str]) -> None:
    _CURRENT_USER_ID.reset(token)


def _user_headers() -> dict[str, str]:
    user_id = _CURRENT_USER_ID.get()
    return {
        "X-user_id": user_id,
        "USERNAME": user_id,
    }


def _is_error_result(result: Any) -> bool:
    return isinstance(result, dict) and result.get("error") is True


def _normalize_backend_category(category: str | None) -> str | None:
    if not category:
        return None

    normalized = category.strip()
    if not normalized:
        return None

    uppercase = normalized.upper()
    if uppercase in _BACKEND_CATEGORIES:
        return uppercase

    lowered = normalized.lower().replace("-", "_")
    return _CATEGORY_ALIASES.get(lowered)


def _infer_backend_category(name: str, category: str | None = None) -> str:
    mapped_category = _normalize_backend_category(category)
    if mapped_category:
        return mapped_category

    lowered_name = name.strip().lower()
    for backend_category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in lowered_name for keyword in keywords):
            return backend_category

    return "OTHER"


# ── Input schemas ─────────────────────────────────────────────────────────────

class GetPantryInput(BaseModel):
    """Optional filters for the pantry list endpoint."""
    category: str | None = Field(
        default=None,
        description=(
            "Optional category filter. Flexible values like dairy, produce, seafood, frozen, "
            "beverages, or backend bucket names are normalized automatically."
        ),
    )
    expiring_within_days: int | None = Field(
        default=None,
        description="Only return items expiring within this many days.",
        ge=0,
        le=365,
    )
    search: str | None = Field(
        default=None,
        description="Free-text search across item names.",
    )


class AddPantryItemInput(BaseModel):
    """Structured payload for adding a single pantry item."""
    name: str = Field(..., description="Ingredient name, e.g. 'whole milk'")
    quantity: float = Field(..., gt=0, description="Numeric amount, e.g. 2.0")
    unit: str = Field(
        ...,
        description="Unit of measure: litre | ml | kg | g | pieces | bunch | pack | etc.",
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional category. Flexible values are mapped to the backend buckets "
            "VEGETABLE | DAIRY | FRUIT | MEAT | GROCERY | SNACK | BEVERAGE | BAKERY | OTHER. "
            "If omitted or unclear, the tool infers the best match from the item name."
        ),
    )
    expiry_date: str | None = Field(
        default=None,
        description="ISO 8601 date string, e.g. '2025-07-15'. Omit if unknown.",
    )
    location: str | None = Field(
        default=None,
        description="Physical storage location: fridge | pantry | freezer",
    )
    notes: str | None = Field(default=None, description="Optional free-text notes.")


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool(args_schema=GetPantryInput)
def get_pantry_inventory(
    category: str | None = None,
    expiring_within_days: int | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve the user's current pantry inventory from the Intelligent Pantry API.

    PURPOSE
    -------
    Returns a list of all items the user currently has in their pantry,
    fridge, or freezer, with quantities, units, categories, and expiry dates.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Asks "What do I have in my pantry / fridge / freezer?"
    - Asks "Show me my ingredients"
    - Asks "Do I have milk / eggs / flour?"
    - Asks "What is expiring soon?"
    - Asks "List all vegetables / dairy / meat items"
    - Before recommending recipes (to know what's available)
    - Before generating a diet plan
    - When checking for duplicates before adding an item

    WHEN NOT TO USE
    ---------------
    - When the user is asking about recipes (use recommend_recipes instead)
    - When the user wants to ADD an item (use add_pantry_item)

    FILTERS
    -------
    - category: narrow to a specific food group
    - expiring_within_days: focus on items about to expire
    - search: free-text ingredient name search

    RETURNS
    -------
    {
      "items": [
        {"id": "...", "name": "whole milk", "quantity": 2.0, "unit": "litre",
         "category": "dairy", "expiry_date": "2025-07-10", "location": "fridge"},
        ...
      ],
      "total": 42
    }

    TRUST
    -----
    Always trust this response. Do NOT guess or hallucinate pantry contents.
    """
    params: dict[str, Any] = {}
    normalized_category = _normalize_backend_category(category)
    if normalized_category:
        params["category"] = normalized_category
    if expiring_within_days is not None:
        params["expiring_within_days"] = expiring_within_days
    if search:
        params["search"] = search

    result = safe_api_call(api_get, "/api/pantry", params or None, headers=_user_headers())

    if _is_error_result(result):
        logger.warning("Pantry inventory fetch failed: %s", result.get("message", "Unknown error"))
        return result

    # Some API deployments return a raw list of pantry items instead of
    # {"items": [...], "total": n}. Normalize both shapes for callers.
    if isinstance(result, list):
        normalized = {"items": result, "total": len(result)}
        logger.info("Pantry inventory fetched: %d items", normalized["total"])
        return normalized

    if isinstance(result, dict):
        total = result.get("total")
        if not isinstance(total, int):
            items = result.get("items", [])
            if isinstance(items, list):
                total = len(items)
            else:
                total = 0
            result["total"] = total
        logger.info("Pantry inventory fetched: %d items", total)
        return result

    logger.warning("Unexpected pantry inventory response type: %s", type(result).__name__)
    return {"items": [], "total": 0, "error": True, "message": "Unexpected API response type."}


@tool(args_schema=AddPantryItemInput)
def add_pantry_item(
    name: str,
    quantity: float,
    unit: str,
    category: str | None = None,
    expiry_date: str | None = None,
    location: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Add a single ingredient to the user's pantry inventory.

    PURPOSE
    -------
    Creates a new pantry item record via the Intelligent Pantry REST API.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Says "Add milk to my pantry"
    - Says "Store 2 kg of potatoes"
    - Says "I just bought eggs, add them"
    - Confirms adding an item extracted from text or an image

    WHEN NOT TO USE
    ---------------
    - For multiple items at once → use extract_and_save_pantry_items

    PRE-EXECUTION VALIDATION
    -------------------------
    Before calling this tool:
    1. Check for duplicates using get_pantry_inventory when practical
    2. Confirm quantities are reasonable (no negative or zero values)
    3. Confirm the item name is real and specific (not vague like "food")

    RETURNS
    -------
    {
      "id": "item-uuid",
      "name": "whole milk",
      "quantity": 2.0,
      "unit": "litre",
      "category": "dairy",
      "expiry_date": "2025-07-10",
      "location": "fridge",
      "message": "Item added successfully."
    }

    ERROR HANDLING
    --------------
    Returns {"error": true, "message": "..."} on failure.
    Do NOT retry silently – surface the error to the user.
    """
    resolved_category = _infer_backend_category(name, category)
    normalized_name = name.strip()
    normalized_unit = unit.strip()

    payload: dict[str, Any] = {
        "name": normalized_name,
        "itemName": normalized_name,
        "quantity": quantity,
        "unit": normalized_unit,
        "category": resolved_category,
    }
    if expiry_date:
        payload["expiry_date"] = expiry_date
    if location:
        payload["location"] = location
    if notes:
        payload["notes"] = notes

    result = safe_api_call(api_post, "/api/pantry", payload, headers=_user_headers())
    if _is_error_result(result):
        logger.warning("Failed to add pantry item %s: %s", name, result.get("message", "Unknown error"))
        return result

    logger.info("Added pantry item: %s (%.1f %s) in category %s", name, quantity, unit, resolved_category)
    return result
