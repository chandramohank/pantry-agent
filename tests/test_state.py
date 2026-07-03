"""Tests for PantryAgentState."""
from __future__ import annotations

from pantry_agent.state import PantryAgentState, default_state


def test_default_state_has_all_keys():
    state = default_state()
    required_keys = [
        "messages", "user_input", "uploaded_images", "intent", "domain",
        "selected_tool", "tool_outputs", "extracted_items", "pantry_items",
        "recipes", "waste_analysis", "sustainability_data", "validation_errors",
        "human_approval_required", "human_approved", "approval_reason",
        "execution_trace", "retry_count", "error", "memory",
    ]
    for key in required_keys:
        assert key in state, f"Missing key: {key}"


def test_default_state_values():
    state = default_state()
    assert state["messages"] == []
    assert state["uploaded_images"] == []
    assert state["human_approval_required"] is False
    assert state["human_approved"] is None
    assert state["retry_count"] == 0
    assert state["error"] is None
    assert isinstance(state["memory"], dict)


def test_state_typeddict_structure():
    """Verify PantryAgentState keys match TypedDict annotations."""
    annotations = PantryAgentState.__annotations__
    state = default_state()
    for key in annotations:
        assert key in state, f"Key '{key}' defined in TypedDict but missing from default_state()"
