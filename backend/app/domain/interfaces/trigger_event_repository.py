from abc import ABC, abstractmethod
from datetime import datetime

from app.domain.models.trigger_event import TriggerEvent


class TriggerEventRepositoryPort(ABC):
    """Port for the append-only trigger/state-change log - see
    `TriggerEvent`'s docstring."""

    @abstractmethod
    def record(self, event: TriggerEvent) -> TriggerEvent:
        """Pure append - never overwrites or dedupes. Deduping "did we
        already log this exact change today" is `daily_close.py`'s own job
        (comparing against yesterday's state before deciding to call this at
        all), not this repository's."""
        ...

    @abstractmethod
    def list_since(self, since: datetime, entity_type: str | None = None) -> list[TriggerEvent]:
        """The raw material Fase 8's trigger-based measurement aggregates
        into hit rates - every event at or after `since`, optionally
        filtered to just "ticker" or "position" events."""
        ...
