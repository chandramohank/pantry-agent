"""
Vision AI Tools
===============
Tools for analyzing food images – scanning groceries, fridge shelves,
pantry shelves, and receipts using computer vision.

PRIORITY RULE: These tools MUST be invoked first whenever the user
supplies an image, regardless of other intent signals.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..api_client import api_post, safe_api_call

logger = logging.getLogger(__name__)


# ── Input schemas ─────────────────────────────────────────────────────────────

class ExtractIngredientsInput(BaseModel):
    """
    Input for extract_ingredients_from_image.
    Provide EITHER image_data (base64) OR image_url – not both.
    """
    image_data: str | None = Field(
        default=None,
        description=(
            "Base64-encoded image string (JPEG / PNG / WebP). "
            "Use this when the image bytes are available in memory."
        ),
    )
    image_url: str | None = Field(
        default=None,
        description=(
            "Publicly accessible or presigned URL pointing to the food image. "
            "Use this when the image is already uploaded to object storage."
        ),
    )
    auto_save: bool = Field(
        default=True,
        description=(
            "When True, detected ingredients are automatically added to the pantry. "
            "Set False to preview results before saving."
        ),
    )


class AskImageInput(BaseModel):
    """Input for ask_about_uploaded_image."""
    image_data: str | None = Field(
        default=None,
        description="Base64-encoded image (JPEG / PNG / WebP).",
    )
    image_url: str | None = Field(
        default=None,
        description="URL of the image to query.",
    )
    question: str = Field(
        ...,
        description=(
            "Natural-language question about the image. "
            "E.g. 'Are these vegetables fresh?' or 'How many tomatoes are there?'"
        ),
    )


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool(args_schema=ExtractIngredientsInput)
def extract_ingredients_from_image(
    image_data: str | None = None,
    image_url: str | None = None,
    auto_save: bool = True,
) -> dict[str, Any]:
    """
    Analyze a food image and extract a structured list of ingredients.

    PURPOSE
    -------
    Uses computer vision to identify ingredients in photos of:
    - Grocery shopping bags or store shelves
    - Open refrigerators or freezers
    - Pantry shelves and cupboards
    - Individual food items or produce
    - Shopping receipts (text extraction + parsing)
    - Kitchen countertops with ingredients laid out

    WHEN TO USE
    -----------
    Use this tool IMMEDIATELY whenever the user:
    - Uploads, attaches, or references an image
    - Says "scan my fridge", "scan my pantry", "scan these groceries"
    - Says "what ingredients do I have?" while providing an image
    - Says "add these groceries" with an attached photo
    - Says "detect items in this image / photo / picture"
    - Says "what vegetables / fruits / proteins are in this image?"
    - Asks to process a shopping receipt photo

    WHEN NOT TO USE
    ---------------
    - Do NOT use for recipe recommendations (use recommend_recipes)
    - Do NOT use for cooking questions (use cooking_copilot)
    - Do NOT use for diet planning (use create_diet_plan)
    - Do NOT use without an image being present

    RETURNS
    -------
    {
      "extracted_items": [
        {"name": "whole milk", "quantity": 2.0, "unit": "litre",
         "category": "dairy", "expiry_date": "2025-07-10"},
        ...
      ],
    "confidence": 0.92,          // 0.0–1.0 confidence score
      "image_description": "...",  // brief description of what was seen
      "saved": true,               // whether items were saved to pantry
      "warnings": []               // low-confidence items or unrecognised objects
    }

    CONFIDENCE HANDLING
    -------------------
    - confidence >= 0.80 → generally reliable extraction
    - confidence < 0.80  → lower-confidence extraction; surface uncertainty in the response

    DEPENDENCIES
    ------------
    None – this is always the first tool called when an image is present.
    """
    if not image_data and not image_url:
        return {"error": True, "message": "Provide either image_data (base64) or image_url."}

    payload: dict[str, Any] = {"auto_save": auto_save}
    if image_data:
        payload["image_data"] = image_data
    if image_url:
        payload["image_url"] = image_url

    result = safe_api_call(api_post, "/api/ai/vision/extract-save", payload)
    logger.info(
        "Vision extraction: %d items, confidence=%.2f",
        len(result.get("extracted_items", [])),
        result.get("confidence", 0),
    )
    return result


@tool(args_schema=AskImageInput)
def ask_about_uploaded_image(
    question: str,
    image_data: str | None = None,
    image_url: str | None = None,
) -> dict[str, Any]:
    """
    Answer a natural-language question about an uploaded food image.

    PURPOSE
    -------
    Lets the user have a conversation about a specific image without
    necessarily extracting or saving ingredients.

    WHEN TO USE
    -----------
    Use this tool when the user has uploaded an image AND is asking a
    specific question about it, such as:
    - "What is this fruit?"
    - "Are these vegetables fresh?"
    - "Which cuisine does this belong to?"
    - "Can I cook pasta with what's shown here?"
    - "How many eggs are in this photo?"
    - "Is the chicken in this image raw or cooked?"
    - "What's the best-before date on this label?"

    WHEN NOT TO USE
    ---------------
    - When the user wants to SAVE ingredients → use extract_ingredients_from_image
    - When there is NO image → use cooking_copilot for general questions

    RETURNS
    -------
    {
      "answer": "The image shows approximately 6 ripe Roma tomatoes...",
      "confidence": 0.88,
      "detected_items": ["tomato", "basil", "garlic"]
    }

    DEPENDENCIES
    ------------
    Requires an image (image_data or image_url).
    """
    if not image_data and not image_url:
        return {"error": True, "message": "Provide either image_data (base64) or image_url."}
    if not question.strip():
        return {"error": True, "message": "A non-empty question is required."}

    payload: dict[str, Any] = {"question": question}
    if image_data:
        payload["image_data"] = image_data
    if image_url:
        payload["image_url"] = image_url

    return safe_api_call(api_post, "/api/ai/vision/ask-image", payload)
