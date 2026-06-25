"""Provider-agnostic embedding client.

Embedders are intentionally separate from LLM clients: many teams want to mix
models (Anthropic for code generation, Voyage for retrieval, Ollama for local
embedding). The factory in `llm.factory.create_llm_client` does not return an
EmbeddingClient on purpose.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class EmbeddedVector:
    text: str
    vector: list[float]


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[EmbeddedVector]:
        return [EmbeddedVector(text=text, vector=self.embed(text)) for text in texts]


class VoyageEmbeddingClient(EmbeddingClient):
    """Voyage AI embedding client (Anthropic's recommended retrieval provider)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "voyage-3",
        base_url: str = "https://api.voyageai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise RuntimeError("VOYAGE_API_KEY is required for VoyageEmbeddingClient")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": text},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return list(response.json()["data"][0]["embedding"])


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbeddingClient")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": text},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return list(response.json()["data"][0]["embedding"])


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/api/embeddings",
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            self._base_url,
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return list(response.json()["embedding"])


def create_embedding_client(embeddings_config: dict | None = None) -> EmbeddingClient | None:
    """Construct an EmbeddingClient from `config["embeddings"]`.

    Returns ``None`` for missing config or unavailable keys; the Qdrant retrieval
    path already degrades gracefully when the embedding client is unavailable.
    """
    config = dict(embeddings_config or {})
    provider = (config.get("provider") or "voyage").lower()
    if provider == "voyage":
        api_key = os.environ.get(config.get("api_key_env", "VOYAGE_API_KEY"))
        if not api_key:
            return None
        return VoyageEmbeddingClient(api_key=api_key, model=config.get("model", "voyage-3"))
    if provider == "openai":
        api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            return None
        return OpenAIEmbeddingClient(api_key=api_key, model=config.get("model", "text-embedding-3-large"))
    if provider == "ollama":
        return OllamaEmbeddingClient(model=config.get("model", "nomic-embed-text"))
    return None
