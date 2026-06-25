"""Unit tests for ``llm.factory.create_llm_client``.

The factory's job is to map config -> concrete ``LLMClient`` for each supported
provider. These tests do not require real API keys — they assert the wiring
paths so a future refactor cannot quietly break Anthropic/Google/OpenAI.
"""
from __future__ import annotations

import pytest

from llm.client import (
    AnthropicLLMClient,
    GeminiLLMClient,
    LLMClient,
    OpenAICompatLLMClient,
)
from llm.factory import create_llm_client


def _clear_env(*names: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_factory_returns_none_when_anthropic_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env("ANTHROPIC_API_KEY", monkeypatch=monkeypatch)
    client = create_llm_client({"provider": "anthropic"})
    assert client is None


def test_factory_builds_anthropic_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = create_llm_client({"provider": "anthropic", "model": "claude-test"})
    assert isinstance(client, LLMClient)
    assert isinstance(client, AnthropicLLMClient)
    assert client.model_name == "claude-test"


def test_factory_routes_openai_provider_to_openai_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    client = create_llm_client({"provider": "openai", "model": "gpt-4o-test"})
    assert isinstance(client, OpenAICompatLLMClient)
    assert client._model == "gpt-4o-test"
    assert client._base_url == "https://api.openai.com/v1"


def test_factory_openai_returns_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env("OPENAI_API_KEY", monkeypatch=monkeypatch)
    assert create_llm_client({"provider": "openai"}) is None


def test_factory_routes_openrouter_with_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    client = create_llm_client({"provider": "openrouter"})
    assert isinstance(client, OpenAICompatLLMClient)
    assert client._base_url.startswith("https://openrouter.ai")


def test_factory_routes_nvidia_nim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    client = create_llm_client({"provider": "nvidia_nim"})
    assert isinstance(client, OpenAICompatLLMClient)
    assert "integrate.api.nvidia.com" in client._base_url


def test_factory_routes_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    client = create_llm_client({"provider": "gemini", "model": "gemini-1.5-pro-test"})
    assert isinstance(client, GeminiLLMClient)
    assert client.model_name == "gemini-1.5-pro-test"


def test_factory_ollama_does_not_require_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env("OLLAMA_API_KEY", monkeypatch=monkeypatch)
    client = create_llm_client({"provider": "ollama", "model": "llama3.1"})
    assert isinstance(client, OpenAICompatLLMClient)
    assert client._base_url.startswith("http://localhost:11434")


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        create_llm_client({"provider": "not_a_real_provider"})


def test_factory_picks_up_custom_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUSTOM_KEY", "custom")
    client = create_llm_client(
        {"provider": "anthropic", "api_key_env": "CUSTOM_KEY", "model": "claude-custom"}
    )
    assert isinstance(client, AnthropicLLMClient)
    assert client.model_name == "claude-custom"
