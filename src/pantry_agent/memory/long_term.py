"""
Long-term memory store
======================
Persists user preferences, pantry history, favourite recipes, and
learned substitutions across sessions using SQLite (or an in-memory dict
during testing / when MEMORY_BACKEND=memory).

Schema (one JSON blob per user_id key):
{
  "dietary_preferences": ["vegetarian", "gluten_free"],
  "allergies": ["nuts"],
  "favourite_recipes": ["Spaghetti Carbonara"],
  "substitutions": {"butter": "coconut oil"},
  "pantry_snapshot": [...],          // last known pantry state
  "waste_patterns": ["often wastes spinach"],
  "session_count": 12,
  "last_updated": "2025-07-03T08:00:00Z"
}
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

# ── In-memory fallback (for testing or MEMORY_BACKEND=memory) ─────────────────
_IN_MEMORY_STORE: dict[str, dict[str, Any]] = {}


# ── SQLite backend ────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    db_path = Path(settings.memory_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id TEXT PRIMARY KEY,
            data    TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_user_memory(user_id: str) -> dict[str, Any]:
    """Load the long-term memory record for *user_id*."""
    if settings.memory_backend == "memory":
        return _IN_MEMORY_STORE.get(user_id, _default_memory())

    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT data FROM user_memory WHERE user_id = ?", (user_id,)
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return _default_memory()
    except Exception as exc:
        logger.error("Failed to load memory for %s: %s", user_id, exc)
        return _default_memory()


def save_user_memory(user_id: str, memory: dict[str, Any]) -> None:
    """Persist the long-term memory record for *user_id*."""
    memory["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
    memory["session_count"] = memory.get("session_count", 0) + 1

    if settings.memory_backend == "memory":
        _IN_MEMORY_STORE[user_id] = memory
        return

    try:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO user_memory (user_id, data, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                data = excluded.data,
                updated_at = excluded.updated_at
            """,
            (user_id, json.dumps(memory), memory["last_updated"]),
        )
        conn.commit()
        conn.close()
        logger.debug("Long-term memory saved for user %s", user_id)
    except Exception as exc:
        logger.error("Failed to save memory for %s: %s", user_id, exc)


def merge_memory_updates(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """
    Merge delta updates into the existing memory dict.

    List fields are unioned (deduplicated). Dict fields are shallow-merged.
    Scalar fields are replaced.
    """
    result = dict(existing)

    list_fields = {"dietary_preferences", "allergies", "favourite_recipes", "waste_patterns"}
    dict_fields = {"substitutions"}

    for key, value in updates.items():
        if key in list_fields:
            current = result.get(key, [])
            merged = list(dict.fromkeys(current + (value or [])))  # deduplicate, preserve order
            result[key] = merged
        elif key in dict_fields:
            current = result.get(key, {})
            result[key] = {**current, **(value or {})}
        else:
            result[key] = value

    return result


def _default_memory() -> dict[str, Any]:
    return {
        "dietary_preferences": [],
        "allergies": [],
        "favourite_recipes": [],
        "substitutions": {},
        "pantry_snapshot": [],
        "waste_patterns": [],
        "session_count": 0,
        "last_updated": None,
    }
