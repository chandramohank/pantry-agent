"""PantryAgentState – the single source of truth flowing through every graph node."""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class PantryAgentState(TypedDict):
    # ── Conversation ─────────────────────────────────────────────────────────
    # `add_messages` reducer appends new messages rather than replacing.
    messages: Annotated[list, add_messages]
    user_input: str

    # ── Vision ───────────────────────────────────────────────────────────────
    # List of base64-encoded image strings or presigned URLs supplied by the user.
    uploaded_images: list[str]

    # ── Classification ───────────────────────────────────────────────────────
    # Canonical intent label, e.g. "scan_groceries" | "get_recipes" | "ask_cooking"
    intent: str
    # Business domain label: Vision | Pantry | Recipes | Diet | Cooking |
    #                         Waste | Sustainability | General
    domain: str

    # ── Tool execution ───────────────────────────────────────────────────────
    selected_tool: str
    tool_outputs: list[dict[str, Any]]

    # ── Domain-specific data surfaces ────────────────────────────────────────
    extracted_items: list[dict[str, Any]]      # items parsed by vision / NLP
    pantry_items: list[dict[str, Any]]          # current pantry inventory
    recipes: list[dict[str, Any]]               # recommended recipes
    waste_analysis: list[dict[str, Any]]        # waste-risk analysis rows
    sustainability_data: dict[str, Any]         # sustainability insights

    # ── Validation & human-in-the-loop ───────────────────────────────────────
    validation_errors: list[str]
    human_approval_required: bool
    human_approved: bool | None   # None = not yet decided
    approval_reason: str          # human-readable explanation for the gate

    # ── Execution metadata ───────────────────────────────────────────────────
    execution_trace: list[dict[str, Any]]
    retry_count: int
    error: str | None

    # ── Memory ───────────────────────────────────────────────────────────────
    # Loaded once at the start of each turn from long-term store.
    memory: dict[str, Any]


def default_state() -> PantryAgentState:
    """Return a fully-initialised blank state (useful for tests and demos)."""
    return PantryAgentState(
        messages=[],
        user_input="",
        uploaded_images=[],
        intent="",
        domain="",
        selected_tool="",
        tool_outputs=[],
        extracted_items=[],
        pantry_items=[],
        recipes=[],
        waste_analysis=[],
        sustainability_data={},
        validation_errors=[],
        human_approval_required=False,
        human_approved=None,
        approval_reason="",
        execution_trace=[],
        retry_count=0,
        error=None,
        memory={},
    )
