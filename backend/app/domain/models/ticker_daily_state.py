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
    # Parte 7 (2026-09, later pass): the ticker-only half of the real design
    # (`trade_geometry.compute_entry_geometry`, via `trade_geometry.geometry_to_dict`)
    # - `None` for rows computed before this column existed, or when the gate
    # itself never got a real ema21/ema55 to build it from. Never sized
    # (`shares_for_risk_budget`/`position_value`/`pct_of_portfolio` are always
    # `None` inside it) - a precomputed universe-wide row doesn't belong to
    # any one portfolio; a caller with a specific portfolio's capital sizes it
    # via `trade_geometry.size_position`/`geometry_from_dict` at read time.
    entry_geometry: dict | None = None
    # Parte 5.3 (2026-09, later pass): `levels_engine.GradeResult` as a plain
    # JSON-safe dict (`{"grade": "A"|"B"|"C"|None, "reasons": [str, ...]}`) -
    # same choice `gate_conditions` above already made, and for the same
    # reason `entry_geometry` never needs a `_from_dict` counterpart: nothing
    # downstream recomputes or resizes a grade at read time, it's a terminal
    # display value. `None` for rows computed before this column existed, or
    # when there was no viable entry geometry to grade in the first place.
    grade: dict | None = None
    # Biblioteca de setups del Radar (en curso, quant_methodology.md §28):
    # lista de `setups.types.SetupMatch` como dicts planos
    # (`setup_match_to_dict`). `[]` es el resultado normal y esperado - "sin
    # coincidencias hoy" para la mayoría de tickers la mayoría de días,
    # mismo criterio que `gate_conditions`. `None` solo para una fila
    # calculada antes de que esta columna existiera.
    setups: list[dict] | None = None
