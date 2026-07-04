"""
AI Extraction Tools
===================
Extract structured pantry items from free-form natural language text.
Bridges user speech / typed shopping lists into structured data.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_post, safe_api_call

logger = logging.getLogger(__name__)


# ── Input schemas ─────────────────────────────────────────────────────────────

class ExtractTextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        description=(
            "Free-form text containing ingredient mentions. "
            "E.g. '2kg onions, 5 tomatoes, a litre of milk and some butter'."
        ),
    )


class ExtractAndSaveInput(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        description=(
            "Natural-language shopping list or ingredient description to extract "
            "AND save directly to the pantry."
        ),
    )


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool(args_schema=ExtractTextInput)
def extract_pantry_items_from_text(text: str) -> dict[str, Any]:
    """
    Parse free-form natural language text and extract structured pantry items.

    PURPOSE
    -------
    Uses AI to identify ingredient names, quantities, and units from
    unstructured text – shopping lists, voice transcripts, recipe ingredient
    sections, or grocery receipts.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Dictates or types a shopping list: "I bought milk, eggs and bread"
    - Provides quantities in mixed formats: "2kg onions and 5 tomatoes"
    - Pastes a grocery receipt text
    - Says "I purchased chicken and yogurt at the store"
    - Wants to PREVIEW what would be extracted BEFORE saving

    WHEN NOT TO USE
    ---------------
    - When an image is provided → use extract_ingredients_from_image
    - When the user explicitly wants to save immediately → use extract_and_save_pantry_items
    - For single items with known details → use add_pantry_item

    RETURNS
    -------
    {
      "extracted_items": [
        {"name": "milk", "quantity": 1.0, "unit": "litre", "category": "dairy"},
        {"name": "eggs", "quantity": 12.0, "unit": "pieces", "category": "dairy"},
        {"name": "bread", "quantity": 1.0, "unit": "loaf", "category": "bakery"}
      ],
      "raw_text": "I bought milk, eggs and bread",
      "confidence": 0.95
    }

    NOTE
    ----
    This tool ONLY extracts – it does NOT save. Call add_pantry_item or
    extract_and_save_pantry_items to persist the results.
    """
    result = safe_api_call(api_post, "/api/ai/extract", {"text": text})
    logger.info(
        "Text extraction: %d items from %d chars",
        len(result.get("extracted_items", [])),
        len(text),
    )
    return result


@tool(args_schema=ExtractAndSaveInput)
def extract_and_save_pantry_items(text: str) -> dict[str, Any]:
    """
    Extract pantry items from natural language text AND save them to the pantry
    in a single API call.

    PURPOSE
    -------
    One-shot operation: NLP parsing + pantry persistence. More efficient than
    chaining extract_pantry_items_from_text → add_pantry_item for each item.

    WHEN TO USE
    -----------
    Use this tool when the user:
    - Says "Add milk, bread, and eggs to my pantry"
    - Provides a text list and wants it saved
    - Pastes a shopping list and says "save all of these"
    - Says "Store everything from this grocery list: ..."

    WHEN NOT TO USE
    ---------------
    - When the user only wants to preview → use extract_pantry_items_from_text
    - When an image is provided → use extract_ingredients_from_image

    RETURNS
    -------
    {
      "extracted_items": [...],  // all items the AI found
      "saved_items": [...],      // items successfully saved
      "skipped_items": [...],    // items skipped (duplicates or low confidence)
      "message": "Saved 3 of 4 items. 1 item was skipped as a duplicate."
    }
    """
    result = safe_api_call(api_post, "/api/ai/extract-and-save", {"text": text})
    logger.info(
        "Extract-and-save: %d saved, %d skipped",
        len(result.get("saved_items", [])),
        len(result.get("skipped_items", [])),
    )
    return result
