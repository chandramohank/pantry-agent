"""
Test configuration and shared fixtures.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Environment setup ─────────────────────────────────────────────────────────
# Override settings before any pantry_agent imports resolve them.

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("PANTRY_API_BASE_URL", "http://localhost:8000")
os.environ.setdefault("MEMORY_BACKEND", "memory")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pantry_items() -> list[dict[str, Any]]:
    return [
        {"id": "1", "name": "whole milk", "quantity": 2.0, "unit": "litre", "category": "dairy", "expiry_date": "2025-07-10"},
        {"id": "2", "name": "eggs", "quantity": 12.0, "unit": "pieces", "category": "dairy", "expiry_date": "2025-07-20"},
        {"id": "3", "name": "spinach", "quantity": 200.0, "unit": "g", "category": "produce", "expiry_date": "2025-07-04"},
        {"id": "4", "name": "chicken breast", "quantity": 500.0, "unit": "g", "category": "meat", "expiry_date": "2025-07-05"},
    ]


@pytest.fixture
def sample_recipes() -> list[dict[str, Any]]:
    return [
        {
            "id": "r1",
            "name": "Spinach Omelette",
            "description": "Quick and healthy breakfast",
            "ingredients": [{"name": "eggs", "quantity": 3, "unit": "pieces"}, {"name": "spinach", "quantity": 50, "unit": "g"}],
            "instructions": ["Beat eggs", "Add spinach", "Cook in pan"],
            "prep_time_minutes": 5,
            "cook_time_minutes": 5,
            "servings": 1,
        }
    ]


@pytest.fixture
def pantry_api_mock():
    """Mock the API client functions to avoid real HTTP calls."""
    with (
        patch("pantry_agent.api_client.api_get") as mock_get,
        patch("pantry_agent.api_client.api_post") as mock_post,
    ):
        yield {"get": mock_get, "post": mock_post}


@pytest.fixture
def mock_openai_response():
    """Return a factory for fake ChatOpenAI responses."""
    def _make_response(content: str = "Test response", tool_calls: list | None = None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls or []
        return msg
    return _make_response
