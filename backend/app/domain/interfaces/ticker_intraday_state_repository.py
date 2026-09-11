from abc import ABC, abstractmethod

from app.domain.models.ticker_intraday_state import TickerIntradayState


class TickerIntradayStateRepositoryPort(ABC):
    """Port for the latest intraday quote/trigger re-check - see
    `TickerIntradayState`'s docstring."""

    @abstractmethod
    def upsert(self, state: TickerIntradayState) -> TickerIntradayState:
        """Overwrites the single row for this ticker in place - there is no
        history here by design, see the domain model's docstring."""
        ...

    @abstractmethod
    def get(self, ticker: str) -> TickerIntradayState | None:
        ...

    @abstractmethod
    def list_by_region(self, region: str) -> list[TickerIntradayState]:
        ...
