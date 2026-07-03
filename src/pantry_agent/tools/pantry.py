"""
Pantry Management Tools
=======================
CRUD operations on the user's personal pantry inventory.
Never hallucinate pantry items – always reflect real API responses.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_get, api_post, safe_api_call

logger = logging.getLogger(__name__)


# ── Input schemas ─────────────────────────────────────────────────────────────

class GetPantryInput(BaseModel):
    """Optional filters for the pantry list endpoint."""
    category: str | None = Field(
        default=None,
        description=(
            "Filter by ingredient category: produce | dairy | meat | seafood | "
            "bakery | frozen | canned | dry_goods | condiments | beverages | snacks | other"
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
    category: str = Field(
        default="other",
        description=(
            "Ingredient category: produce | dairy | meat | seafood | bakery | "
            "frozen | canned | dry_goods | condiments | beverages | snacks | other"
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
    if category:
        params["category"] = category
    if expiring_within_days is not None:
        params["expiring_within_days"] = expiring_within_days
    if search:
        params["search"] = search

    result = safe_api_call(api_get, "/api/pantry", params or None)

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
    category: str = "other",
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
    - When the user has NOT confirmed the addition

    PRE-EXECUTION VALIDATION
    -------------------------
    Before calling this tool:
    1. Check for duplicates using get_pantry_inventory
    2. Confirm quantities are reasonable (no negative or zero values)
    3. For bulk additions (≥5 items), request human approval first
    4. Confirm the item name is real and specific (not vague like "food")

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
    payload: dict[str, Any] = {
        "name": name.strip(),
        "quantity": quantity,
        "unit": unit.strip(),
        "category": category,
    }
    if expiry_date:
        payload["expiry_date"] = expiry_date
    if location:
        payload["location"] = location
    if notes:
        payload["notes"] = notes

    result = safe_api_call(api_post, "/api/pantry", payload)
    logger.info("Added pantry item: %s (%.1f %s)", name, quantity, unit)
    return result
