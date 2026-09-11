from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class PositionDailyState:
    """The precomputed, end-of-day exit_engine read for one open position -
    what `PortfolioRiskService` used to recompute on every cache-miss
    request, now paid once by `daily_close.py` and read cheaply by the "Hoy"
    dashboard (Parte 0, pregunta 1 - "what to do with holdings today"). One
    row per (portfolio_id, ticker, trade_date) - same dated-history reasoning
    as `TickerDailyState`, not just a "latest" cache."""

    id: int | None
    portfolio_id: int
    ticker: str
    trade_date: date
    computed_at: datetime
    urgency: str  # exit_engine.ExitUrgency value
    reasons: list[str]
    price: float
    r_multiple: float | None
    current_stop: float | None
    engine_version: str
