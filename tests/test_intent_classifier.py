"""Tests for intent classification prompt handling."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from pantry_agent.nodes.intent_classifier import classify_intent
from pantry_agent.state import default_state


def test_classify_intent_formats_prompt_without_json_key_error(monkeypatch):
    from pantry_agent.nodes import intent_classifier

    fake_response = MagicMock()
    fake_response.content = json.dumps(
        {
            "intent": "get_pantry",
            "domain": "Pantry",
            "has_image": False,
            "confidence": 0.99,
            "reasoning": "User is asking for pantry contents.",
        }
    )

    def fake_invoke(messages):
        assert len(messages) == 2
        assert messages[0].content.startswith("You are an intent and domain classifier")
        assert '"intent": "<snake_case intent label>"' in messages[0].content
        return fake_response

    fake_llm = MagicMock()
    fake_llm.invoke = fake_invoke
    monkeypatch.setattr(intent_classifier, "_classifier_llm", fake_llm)

    state = default_state()
    state["user_input"] = "get items from pantry"

    result = classify_intent(state)

    assert result["intent"] == "get_pantry"
    assert result["domain"] == "Pantry"
    assert result["uploaded_images"] == []
    assert result["user_input"] == "get items from pantry"
    assert result["execution_trace"][-1]["node"] == "classify_intent"


def test_classify_intent_preferences_override(monkeypatch):
    from pantry_agent.nodes import intent_classifier

    fake_response = MagicMock()
    fake_response.content = json.dumps(
        {
            "intent": "general_query",
            "domain": "General",
            "has_image": False,
            "confidence": 0.9,
            "reasoning": "General request",
        }
    )

    fake_llm = MagicMock()
    fake_llm.invoke = lambda _messages: fake_response
    monkeypatch.setattr(intent_classifier, "_classifier_llm", fake_llm)

    state = default_state()
    state["user_input"] = "show my preferences"

    result = classify_intent(state)

    assert result["intent"] == "get_preferences"
    assert result["domain"] == "General"


def test_classify_intent_add_item_override(monkeypatch):
    from pantry_agent.nodes import intent_classifier

    fake_response = MagicMock()
    fake_response.content = json.dumps(
        {
            "intent": "get_pantry",
            "domain": "Pantry",
            "has_image": False,
            "confidence": 0.9,
            "reasoning": "Pantry-related request.",
        }
    )

    fake_llm = MagicMock()
    fake_llm.invoke = lambda _messages: fake_response
    monkeypatch.setattr(intent_classifier, "_classifier_llm", fake_llm)

    state = default_state()
    state["user_input"] = "Add milk to my pantry"

    result = classify_intent(state)

    assert result["intent"] == "add_item"
    assert result["domain"] == "Pantry"
    assert result["uploaded_images"] == []


def test_classify_intent_location_fridge_does_not_imply_image(monkeypatch):
    from pantry_agent.nodes import intent_classifier

    fake_response = MagicMock()
    fake_response.content = json.dumps(
        {
            "intent": "general_query",
            "domain": "General",
            "has_image": False,
            "confidence": 0.9,
            "reasoning": "General request.",
        }
    )

    fake_llm = MagicMock()
    fake_llm.invoke = lambda _messages: fake_response
    monkeypatch.setattr(intent_classifier, "_classifier_llm", fake_llm)

    state = default_state()
    state["user_input"] = "Add milk, 1 litre, location fridge"

    result = classify_intent(state)

    assert result["intent"] == "add_bulk_items"
    assert result["domain"] == "Pantry"
    assert result["uploaded_images"] == []
