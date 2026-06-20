from __future__ import annotations

from pydantic import BaseModel

from agents.self_update.monitors.khronos_monitor import UpdateDiff
from db.build_queue_store import BuildQueueStore


class ChangelogStore(BaseModel):
    store: BuildQueueStore

    def save(self, diff: UpdateDiff) -> None:
        self.store.save_update_diff(diff)

    def pending_updates(self) -> list[UpdateDiff]:
        return self.store.pending_update_diffs()

    def confirm(self, update_id: str, confirmed: bool) -> UpdateDiff:
        return self.store.confirm_update_diff(update_id, confirmed)
