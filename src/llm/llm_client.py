"""
===============================================================================
llm_client.py
===============================================================================
Provide deterministic and OpenAI-compatible chat clients for curriculum proposals.

Responsibilities:
  - Define the minimal chat-completion interface used by the proposal pipeline
  - Implement an offline deterministic mock client for tests and smoke configs
  - Implement a generic OpenAI-compatible HTTP client backed by requests

Design principles:
  - Keep provider construction config-driven
  - Avoid provider-specific llama.cpp dependencies
  - Never include API keys or bearer tokens in raised error messages

Boundaries:
  - Prompt construction belongs in llm_prompts.py
  - JSON parsing, task validation, and repair policy belong in sibling modules
===============================================================================

"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

DEFAULT_OPENAI_COMPATIBLE_API_BASE = "http://127.0.0.1:18080/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5-coder-32b-instruct-q4_k_m.gguf"
DEFAULT_TIMEOUT_SEC = 60.0
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.0
HTTP_OK = 200
PROVIDER_MOCK = "mock"
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
SUPPORTED_PROVIDERS = (PROVIDER_MOCK, PROVIDER_OPENAI_COMPATIBLE)


class LLMClient(Protocol):
    """Minimal chat-completion protocol used by the LLM curriculum pipeline."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """
        Return a text completion for chat messages.

        Parameters
        ----------
        messages
            Ordered OpenAI-style chat messages with ``role`` and ``content`` keys.

        Returns
        -------
        str
            Provider response content.

        """


class LLMClientError(RuntimeError):
    """Raised when an LLM provider cannot return usable response text."""


@dataclass
class MockLLMClient:
    """
    Deterministic offline LLM client that returns configured responses in order.

    Parameters
    ----------
    responses
        Sequence of raw proposal strings returned one per ``complete`` call.

    """

    responses: Sequence[str]
    _next_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and freeze the configured mock responses."""
        if isinstance(self.responses, str) or not isinstance(self.responses, Sequence):
            message = "mock responses must be a sequence of strings"
            raise TypeError(message)
        if not self.responses:
            message = "mock responses must contain at least one response"
            raise ValueError(message)
        self.responses = tuple(str(response) for response in self.responses)

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """
        Return the next configured mock response.

        Parameters
        ----------
        messages
            Chat messages, accepted for protocol compatibility and ignored.

        Returns
        -------
        str
            Next configured proposal response.

        Raises
        ------
        LLMClientError
            If all configured responses have already been consumed.

        """
        del messages
        if self._next_index >= len(self.responses):
            message = "mock LLM client has no remaining configured responses"
            raise LLMClientError(message)
        response = self.responses[self._next_index]
        self._next_index += 1
        return response


@dataclass(frozen=True)
class OpenAICompatibleLLMClient:
    """
    OpenAI-compatible HTTP chat-completion client using ``requests``.

    Parameters
    ----------
    api_base
        Base URL ending at the OpenAI-compatible ``/v1`` prefix.
    model
        Model identifier sent in the chat-completion payload.
    api_key
        Optional direct API key. Used only when ``api_key_env`` is unset or empty.
    api_key_env
        Optional environment variable name used to read an API key first.
    temperature
        Sampling temperature sent to the provider.
    max_tokens
        Maximum response tokens requested from the provider.
    timeout_sec
        HTTP request timeout in seconds.

    """

    api_base: str = DEFAULT_OPENAI_COMPATIBLE_API_BASE
    model: str = DEFAULT_LOCAL_MODEL
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_sec: float = DEFAULT_TIMEOUT_SEC

    def __post_init__(self) -> None:
        """Validate client settings that can be checked without a network call."""
        if not self.api_base.strip():
            message = "api_base must be non-empty"
            raise ValueError(message)
        if not self.model.strip():
            message = "model must be non-empty"
            raise ValueError(message)
        if self.max_tokens <= 0:
            message = "max_tokens must be positive"
            raise ValueError(message)
        if self.timeout_sec <= 0.0:
            message = "timeout_sec must be positive"
            raise ValueError(message)

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """
        Request one OpenAI-compatible chat completion and return message content.

        Parameters
        ----------
        messages
            Ordered chat messages in OpenAI-compatible format.

        Returns
        -------
        str
            Assistant message content from the first returned choice.

        Raises
        ------
        LLMClientError
            If the provider call fails, returns a non-200 response, or omits
            usable assistant content.

        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _message_payload(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        api_key = self._resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        secrets = self._known_secrets(api_key)
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        except requests.RequestException as exc:
            detail = _redact_text(str(exc), secrets)
            message = f"OpenAI-compatible LLM request failed: {detail}"
            raise LLMClientError(message) from exc

        if response.status_code != HTTP_OK:
            body = _redact_text(response.text[:1000], secrets)
            message = f"OpenAI-compatible LLM request failed with HTTP {response.status_code}: {body}"
            raise LLMClientError(message)

        try:
            decoded = response.json()
        except ValueError as exc:
            body = _redact_text(response.text[:1000], secrets)
            message = f"OpenAI-compatible LLM response was not JSON: {body}"
            raise LLMClientError(message) from exc
        return _extract_chat_content(decoded)

    def _resolved_api_key(self) -> str | None:
        """Return an API key from the configured environment variable or direct config."""
        env_name = (self.api_key_env or "").strip()
        if env_name:
            env_value = os.environ.get(env_name)
            if env_value:
                return env_value
        return self.api_key

    def _known_secrets(self, resolved_api_key: str | None) -> tuple[str, ...]:
        """Return configured secret strings that must not appear in errors."""
        candidates = [resolved_api_key, self.api_key]
        env_name = (self.api_key_env or "").strip()
        if env_name:
            candidates.append(os.environ.get(env_name))
        return tuple(secret for secret in candidates if secret)


def client_from_config(config: Mapping[str, Any]) -> LLMClient:
    """
    Build an LLM client from a provider configuration mapping.

    Parameters
    ----------
    config
        Provider configuration with ``provider`` set to ``mock`` or
        ``openai_compatible``.

    Returns
    -------
    LLMClient
        Configured deterministic or HTTP-backed client.

    Raises
    ------
    ValueError
        If the provider name or required provider fields are invalid.

    """
    provider = str(config.get("provider") or PROVIDER_MOCK)
    if provider == PROVIDER_MOCK:
        return MockLLMClient(_mock_responses_from_config(config))
    if provider == PROVIDER_OPENAI_COMPATIBLE:
        return OpenAICompatibleLLMClient(
            api_base=str(config.get("api_base") or DEFAULT_OPENAI_COMPATIBLE_API_BASE),
            model=str(config.get("model") or DEFAULT_LOCAL_MODEL),
            api_key=_optional_text(config.get("api_key")),
            api_key_env=_optional_text(config.get("api_key_env")),
            temperature=float(config.get("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=int(config.get("max_tokens", DEFAULT_MAX_TOKENS)),
            timeout_sec=float(config.get("timeout_sec", DEFAULT_TIMEOUT_SEC)),
        )
    message = f"llm provider must be one of: {', '.join(SUPPORTED_PROVIDERS)}"
    raise ValueError(message)


def _mock_responses_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return configured mock responses from a provider mapping."""
    raw_responses = config.get("mock_responses", config.get("responses"))
    if isinstance(raw_responses, str) or not isinstance(raw_responses, Sequence):
        message = "mock provider config must contain a mock_responses sequence"
        raise TypeError(message)
    return tuple(str(response) for response in raw_responses)


def _message_payload(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Copy chat messages into JSON-ready dictionaries."""
    payload: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if not role:
            error_message = f"message {index} is missing a role"
            raise LLMClientError(error_message)
        payload.append({"role": role, "content": content})
    return payload


def _extract_chat_content(decoded: Any) -> str:
    """Extract first assistant message content from a provider response."""
    if not isinstance(decoded, Mapping):
        message = "OpenAI-compatible LLM response root must be a mapping"
        raise LLMClientError(message)
    choices = decoded.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        message = "OpenAI-compatible LLM response must contain at least one choice"
        raise LLMClientError(message)
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        message = "OpenAI-compatible LLM choice must be a mapping"
        raise LLMClientError(message)
    message_payload = first_choice.get("message")
    if not isinstance(message_payload, Mapping):
        message = "OpenAI-compatible LLM choice must contain a message mapping"
        raise LLMClientError(message)
    content = message_payload.get("content")
    if not isinstance(content, str) or not content.strip():
        message = "OpenAI-compatible LLM response message content is empty"
        raise LLMClientError(message)
    return content


def _optional_text(value: Any) -> str | None:
    """Return a non-empty string value or ``None``."""
    if value is None:
        return None
    text = str(value)
    return text or None


def _redact_text(text: str, secrets: Sequence[str]) -> str:
    """Return text with known secret values removed."""
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_OPENAI_COMPATIBLE_API_BASE",
    "HTTP_OK",
    "PROVIDER_MOCK",
    "PROVIDER_OPENAI_COMPATIBLE",
    "SUPPORTED_PROVIDERS",
    "LLMClient",
    "LLMClientError",
    "MockLLMClient",
    "OpenAICompatibleLLMClient",
    "client_from_config",
]
