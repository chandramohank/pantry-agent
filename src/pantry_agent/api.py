"""FastAPI app exposing Pantry Agent chat endpoints."""
from __future__ import annotations

import asyncio
from functools import lru_cache
import json
import threading
from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import run_agent
from .models.schemas import AgentResponseEnvelope

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


def _graph_config(request: ChatSSERequest) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }


def _initial_state(request: ChatSSERequest) -> dict[str, Any]:
    return {
        "messages": [("human", request.user_input)],
        "user_input": request.user_input,
        "uploaded_images": list(request.uploaded_images),
    }


def _format_chat_result(request: ChatSSERequest, result: dict[str, Any]) -> dict[str, Any]:
    message = _last_assistant_text(result.get("messages", []))
    ui_response = result.get("ui_response") or {}
    if isinstance(ui_response, dict) and ui_response:
        ui_response = dict(ui_response)
        ui_response["thread_id"] = request.thread_id
        ui_response["message"] = ui_response.get("message") or message
        ui_response.setdefault("payload", {})
        return ui_response

    envelope = AgentResponseEnvelope(
        thread_id=request.thread_id,
        message=message,
        payload={},
        context={
            "intent": result.get("intent", ""),
            "domain": result.get("domain", ""),
        },
        trace=result.get("execution_trace", []),
        errors=result.get("validation_errors", []),
    )
    return envelope.model_dump()


def _run_chat(request: ChatSSERequest) -> dict[str, Any]:
    result = run_agent(
        request.user_input,
        thread_id=request.thread_id,
        user_id=request.user_id,
        uploaded_images=request.uploaded_images,
        app=get_graph_app(),
    )
    return _format_chat_result(request, result)


def _encode_sse_event(event_name: str, payload: Any) -> str:
    data = json.dumps(jsonable_encoder(payload), ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


def _merge_state_update(state: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key == "messages" and isinstance(value, list):
            state.setdefault("messages", []).extend(value)
            continue
        state[key] = value


def _tool_result_content(message: Any) -> Any:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw": content}
    return content


def _chunk_events(request: ChatSSERequest, chunk: Any) -> list[str]:
    if not isinstance(chunk, dict):
        return []

    events: list[str] = []
    for node_name, update in chunk.items():
        if not isinstance(update, dict):
            continue

        for message in update.get("messages", []):
            role = _message_role(message)
            if role in {"ai", "assistant"}:
                for tool_call in getattr(message, "tool_calls", []) or []:
                    events.append(
                        _encode_sse_event(
                            "tool_call",
                            {
                                "thread_id": request.thread_id,
                                "node": node_name,
                                "tool_name": tool_call.get("name"),
                                "arguments": tool_call.get("args", {}),
                                "tool_call_id": tool_call.get("id"),
                            },
                        )
                    )

                text = _message_content_text(getattr(message, "content", ""))
                if text:
                    events.append(
                        _encode_sse_event(
                            "message",
                            {
                                "thread_id": request.thread_id,
                                "node": node_name,
                                "message": text,
                            },
                        )
                    )
            elif role == "tool":
                events.append(
                    _encode_sse_event(
                        "tool_result",
                        {
                            "thread_id": request.thread_id,
                            "node": node_name,
                            "tool_name": getattr(message, "name", None),
                            "tool_call_id": getattr(message, "tool_call_id", None),
                            "result": _tool_result_content(message),
                        },
                    )
                )

        if update.get("error"):
            events.append(
                _encode_sse_event(
                    "error",
                    {
                        "thread_id": request.thread_id,
                        "node": node_name,
                        "message": str(update.get("error")),
                    },
                )
            )

    return events


async def _chat_sse_stream(request: ChatSSERequest):
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    sentinel = object()
    loop = asyncio.get_running_loop()
    state = _initial_state(request)

    def publish(kind: str, payload: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def produce() -> None:
        try:
            graph_app = get_graph_app()
            if hasattr(graph_app, "stream"):
                for chunk in graph_app.stream(
                    state,
                    config=_graph_config(request),
                    stream_mode="updates",
                ):
                    publish("chunk", chunk)
            else:
                publish("final", graph_app.invoke(state, config=_graph_config(request)))
        except Exception as exc:  # pragma: no cover - exercised via API tests
            publish("error", exc)
        finally:
            publish("done", sentinel)

    threading.Thread(target=produce, daemon=True).start()

    yield _encode_sse_event(
        "start",
        {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
            "message": request.user_input,
            "uploaded_images": len(request.uploaded_images),
        },
    )

    stream_completed = False
    while True:
        try:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=15)
        except asyncio.TimeoutError:
            yield _encode_sse_event("keepalive", {"thread_id": request.thread_id})
            continue

        if kind == "done":
            break

        if kind == "error":
            yield _encode_sse_event(
                "error",
                {
                    "thread_id": request.thread_id,
                    "message": str(payload),
                },
            )
            return

        if kind == "final":
            state = payload if isinstance(payload, dict) else state
            final_response = _format_chat_result(request, state)
            yield _encode_sse_event("message", final_response)
            yield _encode_sse_event("done", final_response)
            stream_completed = True
            continue

        if kind == "chunk":
            chunk = payload
            if isinstance(chunk, dict):
                for update in chunk.values():
                    if isinstance(update, dict):
                        _merge_state_update(state, update)
            for event in _chunk_events(request, chunk):
                yield event

    if stream_completed:
        return

    final_response = _format_chat_result(request, state)
    yield _encode_sse_event("done", final_response)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/sse")
async def chat_sse(request: ChatSSERequest) -> StreamingResponse:
    """Run a single chat turn and stream SSE updates as they become available."""
    return StreamingResponse(
        _chat_sse_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
