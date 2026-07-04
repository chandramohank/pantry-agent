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


def test_memory_summary_prompt_formatting_is_valid():
    from pantry_agent.prompts.system_prompts import MEMORY_SUMMARY_PROMPT

    rendered = MEMORY_SUMMARY_PROMPT.format(conversation="User: quick vegan dinner")
    assert "\"substitutions\"" in rendered
    assert "<ingredient>" in rendered


def test_validate_output_collects_tool_data_after_final_ai():
    from langchain_core.messages import ToolMessage

    from pantry_agent.nodes.validator import validate_output

    state = default_state()
    state["messages"] = [
        HumanMessage(content="What can I cook?"),
        AIMessage(content="", tool_calls=[{"name": "recommend_recipes", "args": {}, "id": "tc1"}]),
        ToolMessage(
            content=json.dumps(
                {
                    "recipes": [
                        {
                            "name": "Spinach Omelette",
                            "ingredients": [{"name": "eggs", "quantity": 2, "unit": "pieces"}],
                            "instructions": ["Whisk eggs", "Cook with spinach"],
                        }
                    ]
                }
            ),
            tool_call_id="tc1",
            name="recommend_recipes",
        ),
        AIMessage(content="You can make Spinach Omelette. Whisk eggs and cook with spinach."),
    ]

    result = validate_output(state)

    assert len(result["recipes"]) == 1
    assert len(result["tool_outputs"]) == 1
    assert result["tool_outputs"][0]["tool_name"] == "recommend_recipes"


def test_validate_output_does_not_store_failed_recipe_details():
    from langchain_core.messages import ToolMessage

    from pantry_agent.nodes.validator import validate_output

    state = default_state()
    state["messages"] = [
        HumanMessage(content="Show me the selected recipe"),
        AIMessage(content="", tool_calls=[{"name": "get_recipe_details", "args": {}, "id": "tc1"}]),
        ToolMessage(
            content=json.dumps(
                {
                    "error": True,
                    "message": "Recipe details are unavailable for the selected recipe.",
                }
            ),
            tool_call_id="tc1",
            name="get_recipe_details",
        ),
        AIMessage(content="I couldn't load that recipe just now."),
    ]

    result = validate_output(state)

    assert "recipe_details" not in result
    assert result["validation_errors"] == [
        "get_recipe_details: Recipe details are unavailable for the selected recipe."
    ]


def test_compose_response_summary_message_with_structured_payload():
    from pantry_agent.nodes.response_composer import compose_response

    state = default_state()
    state["messages"] = [
        AIMessage(
            content=(
                "Here is the full recipe: Spinach Omelette. Ingredients: eggs, spinach. "
                "Instructions: whisk eggs, saute spinach, combine and cook."
            )
        )
    ]
    state["recipes"] = [
        {
            "name": "Spinach Omelette",
            "ingredients": [{"name": "eggs", "quantity": 2, "unit": "pieces"}],
            "instructions": ["Whisk eggs", "Cook with spinach"],
        }
    ]

    result = compose_response(state)
    ui_response = result["ui_response"]

    assert ui_response["message"] == "Prepared 1 recipe recommendation(s)."
    assert len(ui_response["payload"]["recipes"]) == 1
    assert "Instructions:" not in ui_response["message"]


def test_compose_response_ignores_failed_recipe_detail_payload():
    from pantry_agent.nodes.response_composer import compose_response

    state = default_state()
    state["messages"] = [AIMessage(content="I couldn't load that recipe just now.")]
    state["recipe_details"] = {
        "error": True,
        "message": "Recipe details are unavailable for the selected recipe.",
    }
    state["validation_errors"] = [
        "get_recipe_details: Recipe details are unavailable for the selected recipe."
    ]

    result = compose_response(state)
    ui_response = result["ui_response"]

    assert ui_response["message"] == "I couldn't load that recipe just now."
    assert "recipe_details" not in ui_response["payload"]
    assert ui_response["artifacts"] == []


def test_compose_response_normalizes_recipe_search_results_into_artifacts():
    from pantry_agent.nodes.response_composer import compose_response

    state = default_state()
    state["messages"] = [AIMessage(content="Here are a few tomato recipes.")]
    state["recipes"] = [
        {
            "id": "recipe-1",
            "title": "Tomato Omelette",
            "url": "https://example.com/tomato-omelette",
            "image": "https://example.com/tomato-omelette.jpg",
            "total_time": 10,
            "calories": 220.5,
            "protein": 12.0,
            "hybrid_score": 0.91,
        }
    ]

    result = compose_response(state)
    ui_response = result["ui_response"]

    assert ui_response["message"] == "Prepared 1 recipe recommendation(s)."
    card = ui_response["artifacts"][0]["data"]["cards"][0]
    assert card["id"] == "recipe-1"
    assert card["name"] == "Tomato Omelette"
    assert card["image_url"] == "https://example.com/tomato-omelette.jpg"
    assert card["prep_time_minutes"] == 10
    assert card["calories"] == 220.5
    assert card["metadata"]["url"] == "https://example.com/tomato-omelette"
    assert card["metadata"]["protein"] == 12.0
    assert card["metadata"]["hybrid_score"] == 0.91
