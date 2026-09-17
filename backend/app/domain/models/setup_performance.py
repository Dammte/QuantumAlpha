from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SetupPerformance:
    """Una fila de la tabla `setup_performance` (Parte 10.2 de la biblioteca
    de setups del Radar) - la medición agregada de un setup, calculada por
    `scripts/setup_replay_study.py` sobre `setup_replay.aggregate_setup_performance`
    y persistida aquí para que `daily_close.py`/el Radar la lean sin
    recalcularla en el propio request.

    `grade`/`market_regime` en `None` significan "todos los grados"/"ambos
    regímenes" (la fila sin segmentar) - no "desconocido". No hay `id`
    autoincremental con significado propio de negocio: esta tabla es una
    FOTO COMPLETA de la última vez que se corrió el estudio, no un
    histórico acumulado - `SetupPerformanceRepository.replace_all` borra
    todo antes de insertar el lote nuevo, así que no hace falta una clave
    única declarada a nivel de base de datos (que además no funcionaría de
    forma fiable con `grade`/`market_regime` en NULL - Postgres trata cada
    NULL como distinto en una restricción UNIQUE)."""

    id: int | None
    setup_name: str
    family: str
    grade: str | None
    market_regime: str | None
    n_observations: int
    trigger_rate: float | None
    win_rate: float | None
    expectancy_r: float | None
    median_bars_held: float | None
    mae_p80_pct: float | None
    failure_rate_3d: float | None
    confidence: str  # "measured" | "thin" | "unvalidated" - SetupConfidence.value
    computed_at: datetime
