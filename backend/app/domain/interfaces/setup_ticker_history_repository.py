from abc import ABC, abstractmethod

from app.domain.models.setup_ticker_history import SetupTickerHistory


class SetupTickerHistoryRepositoryPort(ABC):
    """Port for the persisted per-ticker setup history (Parte 11.2) - see
    `SetupTickerHistory`'s docstring for why this is a full-snapshot table,
    not an incrementally-updated one, mirroring `SetupPerformanceRepositoryPort`."""

    @abstractmethod
    def replace_all(self, rows: list[SetupTickerHistory]) -> None:
        """Borra toda la tabla e inserta `rows` - `scripts/setup_replay_study.py`
        es el único escritor, misma disciplina que `SetupPerformanceRepositoryPort`."""
        ...

    @abstractmethod
    def all(self) -> list[SetupTickerHistory]:
        """Toda la tabla - `GET /market/radar` la lee entera una vez por
        request y la indexa en memoria por (ticker, region, setup_name),
        nunca una consulta por fila."""
        ...
