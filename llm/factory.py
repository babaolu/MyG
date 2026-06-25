"""Factory for `LLMClient` instances configured from `config.yaml["llm"]`.

Supported provider values:

  * ``anthropic``           — Anthropic Claude (default)
  * ``openai``              — OpenAI or OpenAI-compatible Chat Completions
  * ``openrouter``          — OpenRouter aggregation
  * ``nvidia_nim``          — NVIDIA NIM endpoints
  * ``gemini``              — Google Gemini
  * ``ollama``              — Local Ollama (no api_key required)

API keys come from environment variables declared in config (`api_key_env`).
"""
from __future__ import annotations

import os
from typing import Any

from .client import (
    AnthropicLLMClient,
    GeminiLLMClient,
    LLMClient,
    OpenAICompatLLMClient,
)

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-3.5-sonnet",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "nvidia_nim": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-70b-instruct",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "api_key_env": "OLLAMA_API_KEY",  # optional; some local installs ignore it
    },
    "gemini": {
        "model": "gemini-1.5-pro",
        "api_key_env": "GEMINI_API_KEY",
    },
}


def create_llm_client(llm_config: dict[str, Any] | None = None) -> LLMClient | None:
    """Construct an LLMClient from `config["llm"]`.

    Returns ``None`` if the required API key is missing for the chosen
    provider — callers that want to degrade gracefully (e.g. embedding-only or
    rule-based paths) can check truthiness.
    """
    config = dict(llm_config or {})
    provider = (config.get("provider") or "anthropic").lower()
    if provider == "anthropic":
        api_key = os.environ.get(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            return None
        return AnthropicLLMClient(api_key=api_key, model=config.get("model", "claude-sonnet-4-6"))
    if provider in _PROVIDER_DEFAULTS:
        defaults = _PROVIDER_DEFAULTS[provider]
        api_key_env = config.get("api_key_env", defaults["api_key_env"])
        api_key = os.environ.get(api_key_env, "")
        if provider != "ollama" and not api_key:
            return None
        if provider == "gemini":
            if not api_key:
                return None
            return GeminiLLMClient(api_key=api_key, model=config.get("model", defaults["model"]))
        # openai / openrouter / nvidia_nim / ollama — all Chat Completions
        return OpenAICompatLLMClient(
            base_url=config.get("base_url", defaults["base_url"]),
            api_key=api_key or "ollama",  # ollama ignores the bearer token
            model=config.get("model", defaults["model"]),
        )
    raise ValueError(f"unknown llm provider: {provider!r}")
