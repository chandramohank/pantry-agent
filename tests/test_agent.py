"""
Integration tests for the LangGraph agent graph.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from pantry_agent.state import default_state


# ── Validator node tests ──────────────────────────────────────────────────────

class TestValidateOutput:
    def _make_tool_message(self, name: str, content: dict) -> Any:
        from langchain_core.messages import ToolMessage
        return ToolMessage(content=json.dumps(content), tool_call_id="tc1", name=name)

    def test_no_errors_on_clean_output(self):
        from pantry_agent.nodes.validator import validate_output

        state = default_state()
        state["messages"] = [
            HumanMessage(content="What recipes can I make?"),
            AIMessage(content="Here are some recipes."),
        ]
        result = validate_output(state)
        assert result["validation_errors"] == []
        assert result["human_approval_required"] is False

    def test_flags_low_confidence(self):
        from pantry_agent.nodes.validator import validate_output
        from langchain_core.messages import ToolMessage

        tool_msg = ToolMessage(
            content=json.dumps({"extracted_items": [{"name": "milk", "quantity": 1}], "confidence": 0.6}),
            tool_call_id="tc1",
            name="extract_ingredients_from_image",
        )
        state = default_state()
        state["messages"] = [
            HumanMessage(content="scan my fridge"),
            AIMessage(content="", tool_calls=[{"name": "extract_ingredients_from_image", "args": {}, "id": "tc1"}]),
            tool_msg,
        ]
        result = validate_output(state)
        assert result["human_approval_required"] is True
        assert "confidence" in result["approval_reason"].lower()

    def test_flags_bulk_import(self):
        from pantry_agent.nodes.validator import validate_output
        from langchain_core.messages import ToolMessage

        items = [{"name": f"item_{i}", "quantity": 1, "unit": "pieces"} for i in range(10)]
        tool_msg = ToolMessage(
            content=json.dumps({"extracted_items": items, "confidence": 0.95}),
            tool_call_id="tc1",
            name="extract_ingredients_from_image",
        )
        state = default_state()
        state["messages"] = [
            HumanMessage(content="scan my grocery bag"),
            AIMessage(content="", tool_calls=[{"name": "extract_ingredients_from_image", "args": {}, "id": "tc1"}]),
            tool_msg,
        ]
        result = validate_output(state)
        assert result["human_approval_required"] is True

    def test_flags_api_error(self):
        from pantry_agent.nodes.validator import validate_output
        from langchain_core.messages import ToolMessage

        tool_msg = ToolMessage(
            content=json.dumps({"error": True, "message": "API unavailable"}),
            tool_call_id="tc1",
            name="get_pantry_inventory",
        )
        state = default_state()
        state["messages"] = [
            HumanMessage(content="show my pantry"),
            AIMessage(content="", tool_calls=[{"name": "get_pantry_inventory", "args": {}, "id": "tc1"}]),
            tool_msg,
        ]
        result = validate_output(state)
        assert len(result["validation_errors"]) > 0


# ── Memory tests ──────────────────────────────────────────────────────────────

class TestMemory:
    def test_save_and_load_roundtrip(self):
        from pantry_agent.memory.long_term import load_user_memory, save_user_memory

        user_id = "test-user-roundtrip"
        memory = {
            "dietary_preferences": ["vegan"],
            "allergies": ["nuts"],
            "favourite_recipes": ["Pasta"],
            "substitutions": {},
            "pantry_snapshot": [],
            "waste_patterns": [],
            "session_count": 0,
            "last_updated": None,
        }
        save_user_memory(user_id, memory)
        loaded = load_user_memory(user_id)
        assert "vegan" in loaded["dietary_preferences"]
        assert "nuts" in loaded["allergies"]

    def test_merge_memory_updates_unions_lists(self):
        from pantry_agent.memory.long_term import merge_memory_updates

        existing = {
            "dietary_preferences": ["vegetarian"],
            "allergies": [],
            "favourite_recipes": ["Omelette"],
            "substitutions": {},
            "pantry_snapshot": [],
            "waste_patterns": [],
        }
        delta = {
            "dietary_preferences": ["vegan"],
            "favourite_recipes": ["Omelette", "Pasta"],
            "substitutions": {"butter": "coconut oil"},
        }
        merged = merge_memory_updates(existing, delta)
        assert "vegetarian" in merged["dietary_preferences"]
        assert "vegan" in merged["dietary_preferences"]
        assert merged["favourite_recipes"].count("Omelette") == 1  # deduped
        assert "Pasta" in merged["favourite_recipes"]
        assert merged["substitutions"]["butter"] == "coconut oil"

    def test_short_term_recent_turns(self):
        from pantry_agent.memory.short_term import get_recent_turns

        messages = [
            HumanMessage(content="What's in my pantry?"),
            AIMessage(content="You have milk, eggs, and bread."),
            HumanMessage(content="Suggest a recipe."),
            AIMessage(content="Try French toast!"),
        ]
        turns = get_recent_turns(messages, n=4)
        assert len(turns) == 4
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"


# ── Needs-approval routing ────────────────────────────────────────────────────

def test_needs_approval_routing():
    from pantry_agent.nodes.validator import needs_approval

    state_needs = default_state()
    state_needs["human_approval_required"] = True
    assert needs_approval(state_needs) == "request_approval"

    state_skip = default_state()
    state_skip["human_approval_required"] = False
    assert needs_approval(state_skip) == "update_memory"


# ── should_continue routing ───────────────────────────────────────────────────

def test_should_continue_with_tool_calls():
    from pantry_agent.nodes.agent_node import should_continue

    state = default_state()
    ai_msg = AIMessage(content="", tool_calls=[{"name": "get_pantry_inventory", "args": {}, "id": "tc1"}])
    state["messages"] = [ai_msg]
    assert should_continue(state) == "tools"


def test_should_continue_without_tool_calls():
    from pantry_agent.nodes.agent_node import should_continue

    state = default_state()
    ai_msg = AIMessage(content="Here is your pantry summary.")
    state["messages"] = [ai_msg]
    assert should_continue(state) == "validate_output"


# ── Tool registry integration ─────────────────────────────────────────────────

def test_vision_domain_prioritises_vision_tools():
    from pantry_agent.tools.registry import get_tools_for_domain

    tools = get_tools_for_domain("Vision")
    names = [t.name for t in tools]
    assert names[0] == "extract_ingredients_from_image"


def test_all_tools_have_descriptions():
    from pantry_agent.tools.registry import get_all_tools

    for tool in get_all_tools():
        assert tool.description, f"Tool '{tool.name}' has no description"
        assert len(tool.description) > 50, f"Tool '{tool.name}' description is too short"
