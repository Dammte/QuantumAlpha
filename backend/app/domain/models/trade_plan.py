from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class TradePlan:
    """What was proposed (or, for a position opened before this existed, the
    best point-in-time reconstruction of what would have been proposed - see
    `trade_plan_service.py`) when a position was opened: the reference the
    exit engine judges the position against. Before this existed, a stop/
    target was computed once by `build_recommendation`, shown, and never seen
    again - there was nothing durable to compare a held position's current
    price to (see D4 in docs/quant_methodology.md)."""

    id: int | None
    portfolio_id: int
    ticker: str
    entry_price: float
    entry_date: date
    initial_stop: float | None
    initial_target: float | None
    current_stop: float | None  # trailing stop currently in force - starts equal to initial_stop
    highest_close_since_entry: float
    thesis: str
    engine_version: str
    updated_at: datetime
    closed_at: datetime | None  # set once the position that opened this plan is fully sold
