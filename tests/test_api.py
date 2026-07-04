from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pantry_agent import api


class FakeGraph:
    def stream(self, initial_state: dict[str, Any], config: dict[str, Any], stream_mode: str):
        assert initial_state["user_input"] == "What can I cook?"
        assert config["configurable"]["thread_id"] == "thread-1"
        assert stream_mode == "updates"
        yield {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "recipe_search",
                                "args": {"query": "omelette"},
                                "id": "call-1",
                            }
                        ],
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"recipes": ["omelette"]}',
                        name="recipe_search",
                        tool_call_id="call-1",
                    )
                ]
            }
        }
        yield {
            "agent": {
                "messages": [
                    HumanMessage(content="What can I cook?"),
                    AIMessage(content="You can make an omelette."),
                ],
                "intent": "get_recipes",
                "domain": "Recipes",
            }
        }


class BrokenGraph:
    def stream(self, initial_state: dict[str, Any], config: dict[str, Any], stream_mode: str):
        raise RuntimeError("stream exploded")


def _read_sse_events(response) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    current_event: str | None = None
    current_data: str | None = None

    for line in response.iter_lines():
        if not line:
            if current_event and current_data is not None:
                events.append((current_event, json.loads(current_data)))
            current_event = None
            current_data = None
            continue

        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            current_data = line.removeprefix("data: ")

    return events


def test_chat_streams_sse_events(monkeypatch):
    monkeypatch.setattr(api, "get_graph_app", lambda: FakeGraph())
    client = TestClient(api.app)

    with client.stream(
        "POST",
        "/chat/sse",
        json={
            "user_input": "What can I cook?",
            "thread_id": "thread-1",
            "user_id": "user-1",
        },
    ) as response:
        events = _read_sse_events(response)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert [event_name for event_name, _ in events] == [
        "start",
        "tool_call",
        "tool_result",
        "message",
        "done",
    ]

    tool_call_payload = events[1][1]
    assert tool_call_payload["tool_name"] == "recipe_search"
    assert tool_call_payload["arguments"] == {"query": "omelette"}

    tool_result_payload = events[2][1]
    assert tool_result_payload["tool_name"] == "recipe_search"
    assert tool_result_payload["result"] == {"recipes": ["omelette"]}

    message_payload = events[3][1]
    assert message_payload["message"] == "You can make an omelette."

    body = events[4][1]
    assert body["thread_id"] == "thread-1"
    assert body["message"] == "You can make an omelette."
    assert body["payload"] == {}
    assert body["artifacts"] == []
    assert body["schema_version"] == "1.0"
    assert body["context"] == {"intent": "get_recipes", "domain": "Recipes"}


def test_chat_streams_error_event(monkeypatch):
    monkeypatch.setattr(api, "get_graph_app", lambda: BrokenGraph())
    client = TestClient(api.app)

    with client.stream(
        "POST",
        "/chat/sse",
        json={
            "user_input": "What can I cook?",
            "thread_id": "thread-1",
            "user_id": "user-1",
        },
    ) as response:
        events = _read_sse_events(response)

    assert response.status_code == 200
    assert [event_name for event_name, _ in events] == ["start", "error"]
    assert events[1][1] == {
        "thread_id": "thread-1",
        "message": "stream exploded",
    }
