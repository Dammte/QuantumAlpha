from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class TickerDailyState:
    """The precomputed, end-of-day levels/triggers read for one ticker in the
    curated universe - what recomputing `levels_engine.evaluate_gate` on
    demand used to cost per request, now paid once by `daily_close.py`
    (Fase 2's Job A) and read cheaply by every endpoint that needs it (the
    Radar, the ticker deep-dive's summary). `gate_conditions` mirrors
    `levels_engine.GateCondition` as plain JSON-safe dicts
    (`{"label": str, "passed": bool}`), not a rich type - the same choice
    `RecommendationSnapshotORM.factors` already made for its own JSON column.

    One row per (region, ticker, trade_date) - a dated history, not just a
    "latest" cache: the same point-in-time reasoning `UniverseMember`
    established (Segunda auditoría, Bloque 3) applies here too, and Fase 8's
    trigger-based measurement needs to look back at what the gate actually
    said on past dates, not just today's.
    """

    id: int | None
    region: str
    ticker: str
    trade_date: date
    computed_at: datetime
    price: float
    currency: str
    trend: str
    stage: str | None
    rs_rating: int | None
    adx14: float | None
    atr_multiple: float | None
    rsi14: float | None
    gate_passes: bool
    gate_conditions: list[dict]
    gate_version: str
    entry_trigger_type: str | None  # "breakout" | "pullback_bounce" | None
    entry_trigger_price: float | None
    entry_already_triggered: bool
    stop_loss: float | None
    take_profit: float | None
    take_profit_method: str | None
    risk_reward: float | None
