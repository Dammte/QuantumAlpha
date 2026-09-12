from datetime import datetime

from pydantic import BaseModel


class OutcomeStatsResponse(BaseModel):
    """See `signal_performance_service.OutcomeStats` - realized forward-return
    stats for one verdict or position signal at one horizon."""

    label: str  # the verdict ("comprar"/"esperar"/"evitar") or signal string this row summarizes
    horizon_days: int
    n: int
    hit_rate: float | None  # fraction of observations with a positive forward return - not "was the call right"
    mean_return: float | None
    median_return: float | None


class FalseNegativeResponse(BaseModel):
    """See `signal_performance_service.FalseNegative` - a `hold` immediately
    followed by a real drop, named by ticker and date."""

    portfolio_id: int
    ticker: str
    snapshot_at: datetime
    price_at_signal: float
    price_after: float
    return_pct: float
    horizon_days: int


class TriggerOutcomeResponse(BaseModel):
    """See `trigger_performance_service.TriggerOutcomeStats` - Fase 8: the
    new primary "¿está funcionando el sistema?" read, measured against
    `TriggerEvent` (gate/entry-trigger changes) instead of the retired
    checklist's verdict/signal labels above."""

    event_type: str  # "gate_passed" | "entry_triggered"
    horizon_days: int
    n: int
    hit_rate: float | None
    mean_return: float | None
    median_return: float | None


class SignalPerformanceResponse(BaseModel):
    verdict_outcomes: list[OutcomeStatsResponse]
    signal_outcomes: list[OutcomeStatsResponse]
    false_negatives: list[FalseNegativeResponse]
    trigger_outcomes: list[TriggerOutcomeResponse]
    as_of: datetime
