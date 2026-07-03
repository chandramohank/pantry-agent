from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from pantry_agent import api


class FakeGraph:
    def invoke(self, initial_state: dict[str, Any], config: dict[str, Any]):
        assert initial_state["user_input"] == "What can I cook?"
        assert config["configurable"]["thread_id"] == "thread-1"
        return {
            "messages": [
                HumanMessage(content="What can I cook?"),
                AIMessage(content="You can make an omelette."),
            ]
        }


def test_chat_returns_json(monkeypatch):
    monkeypatch.setattr(api, "get_graph_app", lambda: FakeGraph())
    client = TestClient(api.app)

    response = client.post(
        "/chat/sse",
        json={
            "user_input": "What can I cook?",
            "thread_id": "thread-1",
            "user_id": "user-1",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["thread_id"] == "thread-1"
    assert body["message"] == "You can make an omelette."
    assert body["artifacts"] == []
    assert body["schema_version"] == "1.0"
    assert "content" not in body
