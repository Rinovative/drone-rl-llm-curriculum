"""Tests for deterministic and OpenAI-compatible LLM clients."""

# ruff: noqa: S101

from __future__ import annotations

from typing import Any

import pytest

from src import llm

HTTP_ERROR = 500
EXPECTED_TEMPERATURE = 0.2
EXPECTED_MAX_TOKENS = 42
EXPECTED_TIMEOUT_SEC = 3.5
OK_RESPONSE_CONTENT = '{"ok": true}'


class _FakeResponse:
    """Small fake requests response used by client tests."""

    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        """Initialize a fake response."""
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        """Return configured JSON payload or raise like requests.Response.json."""
        if self._payload is None:
            message = "not json"
            raise ValueError(message)
        return self._payload


def test_mock_llm_client_returns_configured_responses_in_order() -> None:
    """Verify the mock client is deterministic and bounded."""
    client = llm.client.MockLLMClient(["first", "second"])

    assert client.complete([]) == "first"
    assert client.complete([]) == "second"
    with pytest.raises(llm.client.LLMClientError, match="no remaining"):
        client.complete([])


def test_client_from_config_builds_mock_provider() -> None:
    """Verify provider construction is config-driven for mock clients."""
    client = llm.client.client_from_config({"provider": "mock", "mock_responses": ["{}"]})

    assert client.complete([]) == "{}"


def test_openai_compatible_client_constructs_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OpenAI-compatible request URL, headers, and payload are constructed without network access."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _FakeResponse:
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse(200, {"choices": [{"message": {"content": OK_RESPONSE_CONTENT}}]})

    monkeypatch.setenv("LOCAL_LLM_KEY", "env-secret")
    monkeypatch.setattr(llm.client.requests, "post", fake_post)
    client = llm.client.OpenAICompatibleLLMClient(
        api_base="http://127.0.0.1:18080/v1/",
        model="local-model",
        api_key="placeholder",
        api_key_env="LOCAL_LLM_KEY",
        temperature=EXPECTED_TEMPERATURE,
        max_tokens=EXPECTED_MAX_TOKENS,
        timeout_sec=EXPECTED_TIMEOUT_SEC,
    )

    content = client.complete([{"role": "user", "content": "hello"}])

    assert content == OK_RESPONSE_CONTENT
    assert captured["url"] == "http://127.0.0.1:18080/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer env-secret"
    assert captured["json"]["model"] == "local-model"
    assert captured["json"]["temperature"] == EXPECTED_TEMPERATURE
    assert captured["json"]["max_tokens"] == EXPECTED_MAX_TOKENS
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["timeout"] == EXPECTED_TIMEOUT_SEC


def test_openai_compatible_client_omits_authorization_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify local providers can run without an Authorization header."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _FakeResponse:
        del url, json, timeout
        captured.update(headers)
        return _FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(llm.client.requests, "post", fake_post)
    client = llm.client.OpenAICompatibleLLMClient(api_base="http://localhost/v1", model="local")

    assert client.complete([{"role": "user", "content": "hello"}]) == "{}"
    assert "Authorization" not in captured


def test_openai_compatible_errors_do_not_leak_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify provider errors redact configured secrets."""
    redaction_value = "token-for-redaction-test"

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _FakeResponse:
        del url, json, headers, timeout
        return _FakeResponse(HTTP_ERROR, text=f"provider rejected {redaction_value}")

    monkeypatch.setattr(llm.client.requests, "post", fake_post)
    client = llm.client.OpenAICompatibleLLMClient(api_base="http://localhost/v1", model="local", api_key=redaction_value)

    with pytest.raises(llm.client.LLMClientError) as exc_info:
        client.complete([{"role": "user", "content": "hello"}])

    assert redaction_value not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)
