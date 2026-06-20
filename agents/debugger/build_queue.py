from __future__ import annotations

import time

from db.build_queue_store import BuildQueueStore
from orchestrator.state import BuildCandidate


def enqueue_candidate(store: BuildQueueStore, source_file: str, cmake_preset: str) -> BuildCandidate:
    candidate = BuildCandidate(id=f"build-{int(time.time() * 1000)}", source_file=source_file, cmake_preset=cmake_preset, created_at=time.time())
    store.add(candidate)
    return candidate


def mark_status(store: BuildQueueStore, candidate_id: str, status: str) -> None:
    store.update_status(candidate_id, status)
