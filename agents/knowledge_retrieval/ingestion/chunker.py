from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_token: int
    end_token: int


def _tokenize(text: str) -> list[str]:
    return text.split()


def chunk_text(text: str, token_size: int = 512, overlap: int = 64) -> list[TextChunk]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    stride = max(token_size - overlap, 1)
    chunks: list[TextChunk] = []
    for start in range(0, len(tokens), stride):
        end = min(start + token_size, len(tokens))
        chunks.append(TextChunk(text=" ".join(tokens[start:end]), start_token=start, end_token=end))
        if end == len(tokens):
            break
    return chunks
