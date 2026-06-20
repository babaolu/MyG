from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from agents.knowledge_retrieval.ingestion.chunker import chunk_text
from orchestrator.state import KnowledgeChunk


@dataclass(frozen=True)
class WebMetadata:
    source_title: str
    platform_tags: list[str]
    topic_tags: list[str]
    vulkan_version: str | None = None
    confidence: str = "vendor_official"


SCRAPER_TARGETS: dict[str, str] = {
    "khronos_spec": "https://registry.khronos.org/vulkan/specs/",
    "khronos_extensions": "https://registry.khronos.org/vulkan/",
    "arm_mali": "https://developer.arm.com/documentation/",
    "qualcomm": "https://developer.qualcomm.com/",
    "jcgt": "https://jcgt.org/published/",
    "advances_rtr": "https://advances.realtimerendering.com/",
}


def scrape_url(url: str, metadata: WebMetadata, timeout: int = 30) -> list[KnowledgeChunk]:
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return [
        KnowledgeChunk(
            id=f"web:{url}:{index}",
            text=chunk.text,
            source_title=metadata.source_title,
            source_type="web",
            platform_tags=metadata.platform_tags,
            topic_tags=metadata.topic_tags,
            vulkan_version=metadata.vulkan_version,
            confidence=metadata.confidence,
            metadata={"url": url},
        )
        for index, chunk in enumerate(chunk_text(text, token_size=512, overlap=64))
    ]


def scrape_known_targets(targets: Iterable[str] = SCRAPER_TARGETS.keys()) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for name in targets:
        base_url = SCRAPER_TARGETS[name]
        metadata = WebMetadata(source_title=name, platform_tags=[], topic_tags=["vulkan"])
        try:
            chunks.extend(scrape_url(base_url, metadata))
        except httpx.HTTPError:
            continue
    return chunks
