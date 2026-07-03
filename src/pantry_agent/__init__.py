"""Pantry Agent – production-ready LangGraph personal kitchen assistant."""

__all__ = ["create_agent", "PantryAgentState", "default_state"]


def __getattr__(name: str):
	if name == "create_agent":
		from .agent import create_agent

		return create_agent
	if name in {"PantryAgentState", "default_state"}:
		from .state import PantryAgentState, default_state

		return {"PantryAgentState": PantryAgentState, "default_state": default_state}[name]
	raise AttributeError(f"module 'pantry_agent' has no attribute {name!r}")
