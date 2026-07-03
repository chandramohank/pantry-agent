"""Tests for configuration helpers."""
from __future__ import annotations

import ssl

from pantry_agent import config


def test_chat_openai_kwargs_uses_ca_bundle(monkeypatch):
    captured: dict[str, dict] = {}
    cafiles: list[str | None] = []

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            captured["sync"] = kwargs

    class FakeAsyncHttpxClient:
        def __init__(self, **kwargs):
            captured["async"] = kwargs

    def fake_create_default_context(*, cafile=None):
        cafiles.append(cafile)
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/Users/ckt7c2a/dev-ca-bundle.pem")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.delenv("AZURE_CA_BUNDLE", raising=False)

    config._get_openai_http_client.cache_clear()
    config._get_openai_async_client.cache_clear()
    monkeypatch.setattr(config.ssl, "create_default_context", fake_create_default_context)
    monkeypatch.setattr(config.httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(config.httpx, "AsyncClient", FakeAsyncHttpxClient)

    kwargs = config.settings.chat_openai_kwargs()

    assert cafiles == ["/Users/ckt7c2a/dev-ca-bundle.pem", "/Users/ckt7c2a/dev-ca-bundle.pem"]
    assert isinstance(captured["sync"]["verify"], ssl.SSLContext)
    assert isinstance(captured["async"]["verify"], ssl.SSLContext)
    assert kwargs["http_client"] is not None
    assert kwargs["http_async_client"] is not None
