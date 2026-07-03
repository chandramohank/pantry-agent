"""
Intent Classifier Node
======================
Uses a lightweight LLM call to:
1. Classify the user's intent into a canonical label
2. Identify the business domain
3. Detect whether an image has been supplied

Runs after load_memory and before the main agent node so routing decisions
are deterministic rather than emergent.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..prompts.system_prompts import INTENT_CLASSIFIER_PROMPT
from ..state import PantryAgentState

logger = logging.getLogger(__name__)

_classifier_llm = ChatOpenAI(
    **settings.chat_openai_kwargs(
        temperature=0.0,   # fully deterministic for classification
        max_tokens=256,
    )
)

_VALID_DOMAINS = {
    "Vision", "Pantry", "Recipes", "Diet", "Cooking",
    "Waste", "Sustainability", "General",
}


def classify_intent(state: PantryAgentState) -> dict[str, Any]:
    """
    Classify user intent and domain; detect image presence.

    Sets: intent, domain, uploaded_images (if images found in messages).
    """
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    # ── Extract the latest human message ──────────────────────────────────
    if not user_input:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                if isinstance(msg.content, str):
                    user_input = msg.content
                elif isinstance(msg.content, list):
                    # Multi-modal content blocks
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_input += block.get("text", "")
                break

    # ── Detect images in message content ──────────────────────────────────
    uploaded_images: list[str] = list(state.get("uploaded_images", []))
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url and url not in uploaded_images:
                        uploaded_images.append(url)

    has_image_hint = bool(uploaded_images)

    # ── Heuristic image keyword detection ────────────────────────────────
    image_keywords = (
        "image", "photo", "picture", "scan", "fridge", "shelf", "receipt",
        "photograph", "snapshot", "pic", "img",
    )
    if any(kw in user_input.lower() for kw in image_keywords):
        has_image_hint = True

    # ── LLM classification ────────────────────────────────────────────────
    prompt = INTENT_CLASSIFIER_PROMPT.format(user_message=user_input or "(empty)")
    try:
        response = _classifier_llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content="Classify now.")]
        )
        raw = response.content if isinstance(response.content, str) else ""

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        classification = json.loads(raw)
    except Exception as exc:
        logger.warning("Intent classification LLM call failed: %s – using defaults", exc)
        classification = {
            "intent": "general_query",
            "domain": "General",
            "has_image": has_image_hint,
            "confidence": 0.5,
        }

    domain = classification.get("domain", "General")
    if domain not in _VALID_DOMAINS:
        domain = "General"

    # Images override domain to Vision if has_image is true
    if classification.get("has_image") or has_image_hint:
        domain = "Vision"
        uploaded_images = uploaded_images or ["<pending>"]

    intent = classification.get("intent", "general_query")

    logger.info("Classified: intent=%s, domain=%s, has_image=%s", intent, domain, bool(uploaded_images))

    return {
        "intent": intent,
        "domain": domain,
        "uploaded_images": uploaded_images,
        "user_input": user_input,
        "execution_trace": state.get("execution_trace", [])
        + [
            {
                "node": "classify_intent",
                "intent": intent,
                "domain": domain,
                "has_image": bool(uploaded_images),
                "classification_confidence": classification.get("confidence", 0),
            }
        ],
    }
