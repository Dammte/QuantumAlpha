from abc import ABC, abstractmethod
from datetime import date

from app.domain.models.ticker_daily_state import TickerDailyState


class TickerDailyStateRepositoryPort(ABC):
    """Port for the precomputed, end-of-day levels/triggers read - see
    `TickerDailyState`'s docstring."""

    @abstractmethod
    def upsert(self, state: TickerDailyState) -> TickerDailyState:
        """Replaces any existing row for this (region, ticker, trade_date) -
        `daily_close.py` re-running on the same day (a manual retry after a
        partial failure) must overwrite, not duplicate. Same delete-then-
        insert idiom `PositionSignalSnapshotRepository.save` already uses."""
        ...

    @abstractmethod
    def get(self, region: str, ticker: str, trade_date: date) -> TickerDailyState | None:
        ...

    @abstractmethod
    def latest_for_ticker(self, ticker: str) -> TickerDailyState | None:
        """The most recent row for one ticker regardless of region - what the
        ticker deep-dive/Radar detail read for a single symbol needs."""
        ...

    @abstractmethod
    def latest_by_region(self, region: str) -> list[TickerDailyState]:
        """Every ticker's most recent row for one region - one call per row's
        own `trade_date` (not necessarily all the same date, if a given
        ticker's latest write happened on an earlier run than the rest) - the
        Radar view's whole data source."""
        ...

    @abstractmethod
    def for_region_and_date(self, region: str, trade_date: date) -> list[TickerDailyState]:
        """Every row for one region on one specific date - `daily_close.py`'s
        own "yesterday vs today" diffing (to detect what changed for
        `TriggerEvent`) needs an exact prior date's snapshot, not just
        "whatever is latest right now"."""
        ...
