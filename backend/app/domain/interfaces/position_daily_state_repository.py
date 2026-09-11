from abc import ABC, abstractmethod
from datetime import date

from app.domain.models.position_daily_state import PositionDailyState


class PositionDailyStateRepositoryPort(ABC):
    """Port for the precomputed, end-of-day exit_engine read on an open
    position - see `PositionDailyState`'s docstring."""

    @abstractmethod
    def upsert(self, state: PositionDailyState) -> PositionDailyState:
        ...

    @abstractmethod
    def latest_for_portfolio(self, portfolio_id: int) -> list[PositionDailyState]:
        """Every held ticker's most recent row for one portfolio - the "Hoy"
        dashboard's whole data source for "what to do with holdings today"."""
        ...

    @abstractmethod
    def for_portfolio_and_date(self, portfolio_id: int, trade_date: date) -> list[PositionDailyState]:
        """Same "exact prior date" need `TickerDailyStateRepositoryPort.
        for_region_and_date` documents, for the position side of
        `daily_close.py`'s own diffing."""
        ...
