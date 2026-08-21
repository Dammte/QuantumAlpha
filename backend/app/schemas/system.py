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


class SignalPerformanceResponse(BaseModel):
    verdict_outcomes: list[OutcomeStatsResponse]
    signal_outcomes: list[OutcomeStatsResponse]
    false_negatives: list[FalseNegativeResponse]
    as_of: datetime


class FactorAblationResultResponse(BaseModel):
    """See `ablation_report_service.FactorAblationResult` - one factor's
    measured effect (from `scripts/factor_ablation_study.py`'s own saved
    output) against its current `recommendation_engine.py` weight, at one
    horizon. `directionally_consistent = False` means the measured sign
    contradicts the weight's sign - flag it, never silently correct it."""

    factor: str
    current_points: int
    mean_difference_pct: float
    directionally_consistent: bool
    significant_at_1pct_bh: bool
    mean_ic: float | None
    ic_ir: float | None
    n_ic_buckets: int
    multivariate_coef_pct: float | None
    multivariate_p_value: float | None


class FactorAblationReportResponse(BaseModel):
    horizon_days: int
    available_horizons_days: list[int]
    results: list[FactorAblationResultResponse]
