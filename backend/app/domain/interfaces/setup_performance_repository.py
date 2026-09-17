from abc import ABC, abstractmethod

from app.domain.models.setup_performance import SetupPerformance


class SetupPerformanceRepositoryPort(ABC):
    """Port for the persisted measurement of the setups library (Parte 10.2)
    - see `SetupPerformance`'s docstring for why this is a full-snapshot
    table, not an incrementally-updated one."""

    @abstractmethod
    def replace_all(self, rows: list[SetupPerformance]) -> None:
        """Borra toda la tabla e inserta `rows` - `scripts/setup_replay_study.py`
        es el único escritor, y cada corrida sustituye por completo la
        medición anterior en vez de acumularla (un estudio re-ejecutado con
        más historial, o tras corregir un detector, debe reemplazar la
        foto vieja, no mezclarse con ella)."""
        ...

    @abstractmethod
    def all(self) -> list[SetupPerformance]:
        """Toda la tabla - `daily_close.py` la lee entera una vez y la indexa
        en memoria por (setup_name, grade, market_regime), igual que ya
        hace con el universo dinámico mensual, en vez de una consulta por
        ticker."""
        ...
