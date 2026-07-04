"""Application settings loaded from environment / .env file."""
from __future__ import annotations

import os
import ssl
from functools import lru_cache
from typing import Any

import httpx
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI / Azure Foundry ──────────────────────────────────────────────
    openai_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
        description="OpenAI/Azure OpenAI API key",
    )
    openai_model: str = Field(default="gpt-5.5")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    azure_openai_endpoint: str = Field(
        default="https://intelligentpantry-ai-foundry.openai.azure.com/openai/v1",
        description="Azure OpenAI endpoint in OpenAI-compatible format",
    )
    foundry_project_endpoint: str = Field(
        default="https://intelligentpantry-ai-foundry.services.ai.azure.com/api/projects/IntelligentPantry-Scan",
        description="Azure AI Foundry project endpoint used for metadata/trace context",
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-ada-002",
        description="Azure OpenAI embedding model deployment name",
    )

    # ── Azure AI Search ──────────────────────────────────────────────────────
    azure_search_endpoint: str = Field(
        default="",
        description="Azure Search service endpoint (e.g., https://myservice.search.windows.net)",
    )
    azure_search_key: str = Field(
        default="",
        description="Azure Search admin key for authentication",
    )
    azure_search_index: str = Field(
        default="recipes-index",
        description="Azure Search index name for recipes",
    )
    azure_search_semantic_config: str = Field(
        default="recipe-semantic-config",
        description="Semantic ranking configuration name in Azure Search index",
    )

    # ── Pantry REST API ──────────────────────────────────────────────────────
    pantry_api_base_url: str = Field(default="http://localhost:8000")
    pantry_api_timeout: float = Field(default=30.0, gt=0)
    pantry_api_key: str | None = Field(default=None)

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_api_key: str | None = Field(default=None)
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="pantry-agent")

    # ── Agent behaviour ──────────────────────────────────────────────────────
    max_iterations: int = Field(default=10, gt=0, le=50)
    vision_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    human_approval_required_for_bulk: int = Field(
        default=5,
        description="Require human approval when adding ≥N items in one operation",
    )

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_backend: str = Field(default="sqlite", pattern="^(sqlite|memory)$")
    memory_db_path: str = Field(default="./data/pantry_memory.db")

    @model_validator(mode="after")
    def _configure_langsmith(self) -> "Settings":
        """Propagate LangSmith config to environment so LangChain picks it up."""
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", self.langchain_api_key)
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_PROJECT", self.langchain_project)
        return self

    def chat_openai_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Shared kwargs for all ChatOpenAI clients in this project."""
        kwargs: dict[str, Any] = {
            "model": self.openai_model,
            "temperature": self.openai_temperature,
            "api_key": self.openai_api_key,
            "base_url": self.azure_openai_endpoint.rstrip("/"),
            # Carry Foundry project endpoint for traceability/debugging.
            "metadata": {"foundry_project_endpoint": self.foundry_project_endpoint},
            "http_client": _get_openai_http_client(),
            "http_async_client": _get_openai_async_client(),
        }
        kwargs.update(overrides)
        return kwargs


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings: Settings = get_settings()


def _resolve_ca_bundle_path() -> str | None:
    """Return the first configured CA bundle path, if any."""
    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "AZURE_CA_BUNDLE"):
        bundle_path = os.environ.get(env_var)
        if bundle_path:
            return bundle_path
    return None


@lru_cache(maxsize=1)
def _get_openai_http_client() -> httpx.Client:
    client_kwargs: dict[str, Any] = {}
    if ssl_context := _build_ssl_context():
        client_kwargs["verify"] = ssl_context
    return httpx.Client(**client_kwargs)


@lru_cache(maxsize=1)
def _get_openai_async_client() -> httpx.AsyncClient:
    client_kwargs: dict[str, Any] = {}
    if ssl_context := _build_ssl_context():
        client_kwargs["verify"] = ssl_context
    return httpx.AsyncClient(**client_kwargs)


def _build_ssl_context() -> ssl.SSLContext | None:
    bundle_path = _resolve_ca_bundle_path()
    if not bundle_path:
        return None
    return ssl.create_default_context(cafile=bundle_path)
