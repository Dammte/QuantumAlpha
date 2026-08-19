from abc import ABC, abstractmethod
from datetime import datetime

from app.domain.models.position_signal_snapshot import PositionSignalSnapshot


class PositionSignalSnapshotRepositoryPort(ABC):
    """Port for persisting position-level signal snapshots (Fase 0
    instrumentation - see `PositionSignalSnapshotORM`'s docstring). Same
    domain/services-stay-framework-free separation as
    `TradePlanRepositoryPort`/`MarketDataProvider`."""

    @abstractmethod
    def save(
        self,
        portfolio_id: int,
        ticker: str,
        signal: str,
        exit_urgency: str | None,
        score: int,
        price: float,
        r_multiple: float | None,
        engine_version: str,
    ) -> PositionSignalSnapshot:
        ...

    @abstractmethod
    def list_all(self, since: datetime | None = None) -> list[PositionSignalSnapshot]:
        """Every snapshot across every portfolio/ticker, optionally only
        those at or after `since` - the raw material
        `signal_performance_service.py` aggregates into hit rates and mean
        returns."""
        ...
