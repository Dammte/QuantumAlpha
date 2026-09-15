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
    # Parte 13: None for the combined row - True/False only for
    # entry_triggered rows split by whether a real BUY followed the trigger
    # (see TriggerOutcomeStats.taken's own docstring).
    taken: bool | None = None


class SignalPerformanceResponse(BaseModel):
    verdict_outcomes: list[OutcomeStatsResponse]
    signal_outcomes: list[OutcomeStatsResponse]
    false_negatives: list[FalseNegativeResponse]
    trigger_outcomes: list[TriggerOutcomeResponse]
    as_of: datetime


class TradingParamsResponse(BaseModel):
    """Reconstruction (2026-09), Parte 19: a read-only mirror of
    `app.core.trading_params` - every field name matches that module's
    constant name exactly, so the UI can show which config produced a given
    decision without a separate translation table to keep in sync."""

    risk_per_trade_pct: float
    max_position_pct: float
    max_aggregate_risk_pct: float
    max_open_positions: int
    min_position_usd: float
    min_position_for_scaling: float
    transaction_cost_pct: float

    stop_atr_ceiling: float
    risk_ceiling_atr_multiple: float
    risk_ceiling_min_pct: float
    risk_ceiling_max_pct: float
    min_risk_reward_net: float

    scale_out_1r_fraction: float
    scale_out_2r_fraction: float
    last_tranche_time_stop_bars: int

    chandelier_window: int
    chandelier_mult_by_vol: dict[str, float]
    chandelier_profit_lock_r: float
    chandelier_profit_lock_mult: float

    trigger_max_distance_atr: float
    breakout_min_rel_volume: float
    stall_min_bars: int
    stall_max_bars: int
    high_correlation_threshold: float
