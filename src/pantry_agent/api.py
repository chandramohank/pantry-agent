"""FastAPI app exposing Pantry Agent chat endpoints."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import run_agent

app = FastAPI(title="Pantry Agent API", version="0.1.0")


class ChatSSERequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    thread_id: str = Field(default="default")
    user_id: str = Field(default="default_user")
    uploaded_images: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_graph_app() -> Any:
    """Lazily initialize the compiled LangGraph app once per process."""
    from .agent import create_agent

    return create_agent()


def _message_role(message: Any) -> str | None:
    msg_type = getattr(message, "type", None)
    if isinstance(msg_type, str):
        return msg_type
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role
    return None


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _last_assistant_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = _message_role(message)
        if role in {"ai", "assistant"}:
            return _message_content_text(getattr(message, "content", ""))
    return ""


def _run_chat(request: ChatSSERequest) -> dict[str, Any]:
    result = run_agent(
        request.user_input,
        thread_id=request.thread_id,
        user_id=request.user_id,
        uploaded_images=request.uploaded_images,
        app=get_graph_app(),
    )
    return {
        "thread_id": request.thread_id,
        "content": _last_assistant_text(result.get("messages", [])),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/sse")
def chat_sse(request: ChatSSERequest) -> dict[str, Any]:
    """Run a single chat turn and return the final response immediately."""
    return _run_chat(request)
