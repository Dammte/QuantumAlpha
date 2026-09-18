from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class JobRun:
    """One execution of a Fase 2 cron job (`daily_close.py`,
    `intraday_refresh.py`, `refresh_universe_membership.py`) - the
    operational record answering "did last night's job actually run, and
    what happened" without trawling Render's own log retention window.
    `rows_processed` and `error_message` are the two things worth checking
    after a silent failure: how much progress was made (some tickers/
    positions updated, others not) and why it stopped."""

    id: int | None
    job_name: str
    started_at: datetime
    finished_at: datetime | None
    status: str  # "running" | "success" | "failed"
    rows_processed: int
    error_message: str | None
    # Resumen ejecutable del run (Auditoria del Radar, bloque C.1) - JSON
    # libre, no un esquema fijo: `daily_close.py` guarda tickers procesados/
    # fallidos (agrupados por tipo de error), cuantos pasan el gate, setups
    # por familia y duracion; otros jobs pueden guardar lo que les sea util
    # sin migrar este modelo cada vez. `None` para runs anteriores a este
    # campo, o para jobs que no necesitan detalle estructurado.
    detail: dict | None = None
