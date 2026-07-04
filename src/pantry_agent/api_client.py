"""Shared synchronous httpx client with retry logic for the Pantry REST API."""
from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import settings

logger = logging.getLogger(__name__)

# Retry on transient network errors only – never on 4xx responses.
_RETRY_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def _build_client() -> httpx.Client:
    headers: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.pantry_api_key:
        headers["X-API-Key"] = settings.pantry_api_key
    return httpx.Client(
        base_url=settings.pantry_api_base_url,
        headers=headers,
        timeout=settings.pantry_api_timeout,
        follow_redirects=True,
    )


# Module-level singleton – re-created if the settings change (e.g., in tests).
_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    global _client  # noqa: PLW0603
    if _client is None or _client.is_closed:
        _client = _build_client()
    return _client


def reset_client() -> None:
    """Force a new client on next use – useful in tests or after config changes."""
    global _client  # noqa: PLW0603
    if _client and not _client.is_closed:
        _client.close()
    _client = None


def _merge_request_headers(headers: dict[str, str] | None = None) -> dict[str, str] | None:
    if not headers:
        return None

    merged = dict(get_client().headers)
    merged.update(headers)
    return merged


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def api_get(
    path: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """GET request to the Pantry API, with automatic retry on transient errors."""
    response = get_client().get(path, params=params, headers=_merge_request_headers(headers))
    response.raise_for_status()
    return response.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def api_post(
    path: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST request to the Pantry API, with automatic retry on transient errors."""
    response = get_client().post(path, json=body or {}, headers=_merge_request_headers(headers))
    response.raise_for_status()
    return response.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def api_delete(
    path: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """DELETE request to the Pantry API."""
    response = get_client().delete(path, headers=_merge_request_headers(headers))
    response.raise_for_status()
    return response.json()


def safe_api_call(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Execute an API call and normalise errors into a structured error dict
    so tool callers always get a consistent return type.
    """
    try:
        return fn(*args, **kwargs)
    except httpx.HTTPStatusError as exc:
        logger.error("Pantry API HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return {
            "error": True,
            "status_code": exc.response.status_code,
            "message": exc.response.text,
        }
    except Exception as exc:
        logger.error("Pantry API call failed: %s", exc)
        return {"error": True, "message": str(exc)}
