"""Provider-agnostic LLM client interface for VulkanMind.

`LLMClient` exposes a single async-friendly surface consumed by every agent and
self-improvement layer. Concrete providers translate that surface into their
native SDK calls. Adding a provider means implementing this interface and
adding a branch in `llm.factory.create_llm_client`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredCompletionError(RuntimeError):
    """Raised when a structured completion cannot be coerced to ``response_model``."""

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract interface for synchronous structured LLM completion.

    Implementations must:
      * Return text shaped to satisfy ``response_model.model_validate_json``
      * Reject responses the provider cannot parse (raise ``StructuredCompletionError``)
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | T: ...

    def complete_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[T],
        *,
        max_tokens: int = 2048,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Complete + validate against a Pydantic model. Implementations may
        override for native JSON-mode enforcement.

        Default flow:
          1. request completion with a schema hint in the user prompt
          2. parse JSON out of the response
          3. model_validate_json
        Falls back to ``{error, raw}`` model construction on ValidationError,
        mirroring the legacy `call_claude_structured` behaviour, then raises
        `StructuredCompletionError` if even that fails.
        """
        schema_hint = (
            "Return only valid JSON matching this Pydantic schema:\n"
            f"{response_model.model_json_schema()}"
        )
        augmented = list(messages)
        if augmented and augmented[0].role == "system":
            augmented[0] = LLMMessage(role="system", content=f"{augmented[0].content}\n\n{schema_hint}")
        else:
            augmented.insert(0, LLMMessage(role="system", content=schema_hint))
        response = self.complete(
            augmented,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        text = _extract_json(response.text if isinstance(response, LLMResponse) else str(response))
        try:
            return response_model.model_validate_json(text)
        except ValidationError as exc:
            try:
                return response_model.model_validate(
                    {"error": str(exc), "raw": text},
                )
            except ValidationError:
                raise StructuredCompletionError(str(exc), raw=text) from exc


def _extract_json(text: str) -> str:
    """Pull the first JSON object from a (possibly fenced) completion."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Strip ```json ... ``` fences
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1]


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class AnthropicLLMClient(LLMClient):
    """LLMClient backed by the Anthropic SDK (claude-sonnet-4-6 family)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        client: Any | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for AnthropicLLMClient")
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | T:
        # Anthropic needs system as a top-level param
        system_text = ""
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system":
                system_text = (system_text + "\n\n" + message.content).strip() if system_text else message.content
            else:
                anthropic_messages.append({"role": message.role, "content": message.content})
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system_text,
            "messages": anthropic_messages,
        }
        if temperature is not None:
            params["temperature"] = temperature
        params.update(kwargs)
        response = self._client.messages.create(**params)
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ) or getattr(response.content[0], "text", "")
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
        if response_model is not None:
            return self.complete_structured(
                messages,
                response_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return LLMResponse(text=text, model=self._model, raw=response.model_dump() if hasattr(response, "model_dump") else {}, usage=usage)


class OpenAICompatLLMClient(LLMClient):
    """LLMClient for OpenAI and OpenAI-compatible HTTP endpoints
    (``api.openai.com``, ``openrouter.ai``, ``integrate.api.nvidia.com``,
    local Ollama, etc.). Activated by any provider value the factory recognises
    whose transport is OpenAI Chat Completions JSON."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | T:
        import httpx

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        # OpenRouter/OpenAI supports response_format json_object when structured
        if response_model is not None:
            payload.setdefault("response_format", {"type": "json_object"})
        payload.update({k: v for k, v in kwargs.items() if k != "response_format"})
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"] if data.get("choices") else ""
        usage = data.get("usage", {}) or {}
        llm_response = LLMResponse(text=text, model=self._model, raw=data, usage=usage)
        if response_model is not None:
            return self.complete_structured(
                messages,
                response_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return llm_response


class GeminiLLMClient(LLMClient):
    """LLMClient for Google Gemini via its `generateContent` REST endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-1.5-pro",
        timeout: float = 60.0,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float | None = None,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | T:
        import httpx

        # Gemini splits system instruction from user/author turns
        system_instruction = None
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                system_instruction = {"parts": [{"text": message.content}]}
            else:
                role = "user" if message.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": message.content}]})
        generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            generation_config["temperature"] = temperature
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        if response_model is not None:
            generation_config["responseMimeType"] = "application/json"
        url = f"{self._base_url}/models/{self._model}:generateContent?key={self._api_key}"
        response = httpx.post(url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
        usage_meta = data.get("usageMetadata", {}) or {}
        usage = {
            "input_tokens": usage_meta.get("promptTokenCount", 0),
            "output_tokens": usage_meta.get("candidatesTokenCount", 0),
        }
        llm_response = LLMResponse(text=text, model=self._model, raw=data, usage=usage)
        if response_model is not None:
            return self.complete_structured(
                messages,
                response_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return llm_response
