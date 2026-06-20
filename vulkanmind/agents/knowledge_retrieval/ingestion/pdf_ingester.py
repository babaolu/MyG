from __future__ import annotations

from pathlib import Path

from llama_parse import LlamaParse
from pydantic import BaseModel, Field

from agents.knowledge_retrieval.ingestion.chunker import chunk_text
from orchestrator.state import KnowledgeChunk


class PdfIngestionResult(BaseModel):
    chunks: list[KnowledgeChunk]


class PdfMetadata(BaseModel):
    source_title: str
    author: str | None = None
    year: int | None = None
    chapter: str | None = None
    platform_tags: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    vulkan_version: str | None = None
    confidence: str = "spec"


def ingest_pdf(path: str | Path, metadata: PdfMetadata, api_key: str | None = None) -> PdfIngestionResult:
    parser = LlamaParse(api_key=api_key) if api_key else LlamaParse(result_type="text")
    documents = parser.load_data(str(path))
    text = "\n".join(document.text for document in documents)
    chunks = [
        KnowledgeChunk(
            id=f"pdf:{path}:{index}",
            text=chunk.text,
            source_title=metadata.source_title,
            source_type="pdf",
            platform_tags=metadata.platform_tags,
            topic_tags=metadata.topic_tags,
            vulkan_version=metadata.vulkan_version,
            confidence=metadata.confidence,
            metadata={
                "author": metadata.author,
                "year": metadata.year,
                "chapter": metadata.chapter,
                "source_path": str(path),
            },
        )
        for index, chunk in enumerate(chunk_text(text, token_size=512, overlap=64))
    ]
    return PdfIngestionResult(chunks=chunks)
