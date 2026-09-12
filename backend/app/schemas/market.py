from datetime import date, datetime

from pydantic import BaseModel

from app.schemas.common import PriceLevelResponse
from app.schemas.quant_analysis import (
    CoreSignalsResponse,
    EntryTriggerResponse,
    GateConditionResponse,
    ImminentCrossResponse,
    MultiTimeframeResponse,
    StopAndTargetResponse,
)


class NewsArticleResponse(BaseModel):
    title: str
    publisher: str | None
    link: str | None
    published_at: str | None


class TickerSnapshotResponse(BaseModel):
    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    currency: str
    price: float
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_6m: float | None
    change_1y: float | None
    volume: float
    relative_volume: float | None
    rsi14: float | None
    sma20: float | None
    sma50: float | None
    sma150: float | None
    sma200: float | None
    dist_52w_high: float | None
    dist_52w_low: float | None
    atr_multiple: float | None
    adx14: float | None
    plus_di: float | None
    minus_di: float | None
    mansfield_rs: float | None
    rs_rating: int | None
    trend: str
    stage: str | None
    ma_cross: str | None
    minervini_score: int
    minervini_pass: bool
    ma_cross_short: str | None = None
    imminent_cross_short_term: ImminentCrossResponse | None = None


class MoversResponse(BaseModel):
    gainers: list[TickerSnapshotResponse]
    losers: list[TickerSnapshotResponse]
    near_52w_high: list[TickerSnapshotResponse]
    near_52w_low: list[TickerSnapshotResponse]
    high_volume: list[TickerSnapshotResponse]
    oversold: list[TickerSnapshotResponse]
    overbought: list[TickerSnapshotResponse]
    golden_cross: list[TickerSnapshotResponse]
    death_cross: list[TickerSnapshotResponse]
    rs_leaders: list[TickerSnapshotResponse]
    strong_trend: list[TickerSnapshotResponse]


class TrendBreadthResponse(BaseModel):
    total: int
    pct_above_sma50: float
    pct_above_sma200: float
    count_uptrend: int
    count_downtrend: int
    count_sideways: int
    golden_crosses: int
    death_crosses: int
    count_overbought: int
    count_oversold: int
    count_stage2: int
    count_minervini_pass: int
    count_imminent_golden: int = 0
    count_imminent_death: int = 0


class TrendDetailResponse(BaseModel):
    uptrend: list[TickerSnapshotResponse]
    downtrend: list[TickerSnapshotResponse]
    golden_cross: list[TickerSnapshotResponse]
    death_cross: list[TickerSnapshotResponse]
    overbought: list[TickerSnapshotResponse]
    oversold: list[TickerSnapshotResponse]
    stage2: list[TickerSnapshotResponse]
    stage4: list[TickerSnapshotResponse]
    minervini_pass: list[TickerSnapshotResponse]
    strong_trend: list[TickerSnapshotResponse]
    imminent_cross: list[TickerSnapshotResponse] = []


class SupportResistanceResponse(BaseModel):
    ticker: str
    currency: str
    price: float
    levels: list[PriceLevelResponse]


class ProximityItemResponse(BaseModel):
    ticker: str
    sector: str
    currency: str
    price: float
    level: PriceLevelResponse


class IndustryUniverseResponse(BaseModel):
    name: str
    sector: str
    etf: str | None
    tickers: list[str]


class UniverseResponse(BaseModel):
    sectors: dict[str, list[str]]  # sector -> industry names
    industries: list[IndustryUniverseResponse]


class IndexSnapshotResponse(BaseModel):
    name: str
    ticker: str
    price: float | None
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_1y: float | None
    trend: str | None


class VixSnapshotResponse(BaseModel):
    level: float | None
    sma50: float | None
    regime: str
    term_structure: str | None


class MarketRegimeResponse(BaseModel):
    verdict: str
    headline: str
    reasons: list[str]


class MarketContextResponse(BaseModel):
    indices: list[IndexSnapshotResponse]
    vix: VixSnapshotResponse
    regime: MarketRegimeResponse
    news: list[NewsArticleResponse]


class TradePlanResponse(BaseModel):
    """See `domain.models.trade_plan.TradePlan` - the persisted (or, for a
    position opened before this existed, point-in-time reconstructed) stop/
    target/thesis the exit engine judges this position against. `None` when
    there's no currently-open lot to plan for (see `trade_plan_service.py`)."""

    entry_price: float
    entry_date: date
    initial_stop: float | None
    initial_target: float | None
    current_stop: float | None
    highest_close_since_entry: float
    thesis: str
    engine_version: str


class ScaledExitPlanResponse(BaseModel):
    """See `trade_manager.ScaledExitPlan` - a suggested (never automatically
    executed) partial exit at the +1R/+2R milestones, in concrete share
    quantities."""

    action: str  # "none" | "sell_at_1r" | "sell_at_2r"
    shares_to_sell: float
    shares_remaining_after: float
    suggested_new_stop: float | None
    description: str


class PositionRiskResponse(BaseModel):
    ticker: str
    currency: str
    price: float
    trend: str
    stage: str | None
    ma_cross: str | None
    rs_rating: int | None
    nearest_support: PriceLevelResponse | None
    nearest_resistance: PriceLevelResponse | None
    signal: str
    score: int
    reasons: list[str]
    signals: CoreSignalsResponse
    # Added for the independent exit engine (see exit_engine.py) - all
    # additive, `signal`/`score`/`reasons` above are unchanged and still mean
    # exactly what they meant before this existed.
    exit_urgency: str | None  # "exit_now"|"reduce"|"tighten_stop"|"watch"|"hold" - None if no open trade plan
    exit_reasons: list[str]
    trade_plan: TradePlanResponse | None
    r_multiple: float | None
    multi_timeframe: MultiTimeframeResponse | None
    scaled_exit: ScaledExitPlanResponse | None
    bars_held: int | None  # closed daily bars since trade_plan.entry_date - "sesiones mantenidas" in the UI


class PortfolioRiskResponse(BaseModel):
    positions: list[PositionRiskResponse]
    computed_at: datetime  # when this was last actually computed - may be hours old, see durable_cache.py


class PositionDailyStateResponse(BaseModel):
    """Reconstruction (2026-09), Fase 5: one row of `PositionDailyStateORM`,
    read as-is - the precomputed counterpart of `PositionRiskResponse` above,
    written once a day by `daily_close.py` instead of recomputed (and
    durably cached) per request. Deliberately narrower than
    `PositionRiskResponse` - no `signals`/`multi_timeframe`/`scaled_exit` -
    this is the fast "what needs attention today" read (Parte 0, pregunta 1);
    the richer live read at `/portfolios/{id}/risk` is unchanged and still
    the one the position detail card opens into."""

    ticker: str
    trade_date: date
    computed_at: datetime
    urgency: str  # "exit_now" | "reduce" | "tighten_stop" | "watch" | "hold"
    reasons: list[str]
    price: float
    r_multiple: float | None
    current_stop: float | None


class DailyBriefResponse(BaseModel):
    brief_date: date
    computed_at: datetime
    positions_needing_action: int
    new_entry_triggers: int
    new_gate_passes: int
    headline: str


class OpportunityCostNoteResponse(BaseModel):
    """Reconstruction (2026-09), Fase 5 (resto): see `opportunity_cost.py`'s
    own docstring - a holding whose own gate doesn't pass today, next to a
    same-sector Radar candidate whose gate does. Never a second opinion on
    whether to sell `held_ticker` (that's exit_engine.py's job alone)."""

    held_ticker: str
    sector: str
    alternative_ticker: str
    alternative_rs_rating: int | None


class PortfolioTodayResponse(BaseModel):
    # `None` when daily_close.py hasn't run for this portfolio yet - same
    # "empty vs. doesn't exist yet" distinction RadarResponse.computed_at
    # documents.
    brief: DailyBriefResponse | None
    positions: list[PositionDailyStateResponse]
    opportunity_cost: list[OpportunityCostNoteResponse]


class RadarItemResponse(BaseModel):
    """Reconstruction (2026-09), Fase 5: one row of `TickerDailyStateORM`,
    read as-is - no live computation behind this response at all, unlike
    every field above it in this file. See `docs/quant_methodology.md` §25
    and `ticker_daily_state.py`'s own docstring."""

    ticker: str
    region: str
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
    gate_conditions: list[GateConditionResponse]
    gate_version: str
    entry_trigger: EntryTriggerResponse | None
    stop_and_target: StopAndTargetResponse | None


class RadarResponse(BaseModel):
    items: list[RadarItemResponse]
    # `None` when `daily_close.py` hasn't run for this region yet (a fresh
    # deploy, or the cron job not yet enabled - see render.yaml) - an empty
    # Radar and a Radar that simply doesn't exist yet are different states,
    # same "prefiero una lista vacía a datos a medias" rule CLAUDE.md already
    # applies elsewhere.
    computed_at: datetime | None


class CorrelationWarningResponse(BaseModel):
    """See portfolio_construction_service.CorrelationWarning - two held
    tickers correlated at or beyond the threshold: one bet with two names,
    not two independent ones."""

    ticker_a: str
    ticker_b: str
    correlation: float


class SectorConcentrationResponse(BaseModel):
    sector: str
    weight_pct: float
    tickers: list[str]


class PositionRiskContributionResponse(BaseModel):
    ticker: str
    weight_pct: float
    risk_contribution_pct: float


class AggregateRiskReportResponse(BaseModel):
    total_risk_amount: float
    total_risk_pct_of_capital: float | None
    exceeds_limit: bool


class PortfolioConstructionResponse(BaseModel):
    """See portfolio_construction_service.py (D12) - portfolio-level risk no
    single position's own analysis can see: correlated bets, sector
    concentration, and the real money lost if every stop got hit at once.
    Positions with no return history yet (a brand-new listing) or no open
    trade plan (aggregate_risk only) are simply absent from the relevant
    section rather than assumed - see each pure function's own docstring."""

    # ticker -> ticker -> correlation, only tickers with return history; None
    # for a pair whose correlation can't be computed (e.g. one side has zero
    # variance over the window) rather than a silently wrong number.
    correlation_matrix: dict[str, dict[str, float | None]]
    correlated_pairs: list[CorrelationWarningResponse]
    sector_concentrations: list[SectorConcentrationResponse]
    concentrated_sectors: list[SectorConcentrationResponse]
    risk_contributions: list[PositionRiskContributionResponse]
    portfolio_volatility_pct: float | None  # annualized realized vol of the portfolio as a whole
    volatility_target_pct: float
    suggested_to_trim: list[PositionRiskContributionResponse]  # top contributors to trim if vol exceeds target
    aggregate_risk: AggregateRiskReportResponse
    tickers_without_trade_plan: list[str]  # excluded from aggregate_risk - no known stop to size risk against
    computed_at: datetime


# --- Relationship map (Tercera auditoría, Bloque G) --------------------------


class StatisticalRelationResponse(BaseModel):
    ticker: str
    sector: str | None
    correlation_60d: float | None
    correlation_250d: float | None
    relative_beta: float | None
    lead_lag_days: int | None
    lead_lag_correlation: float | None
    comovement_extreme_days_pct: float | None
    is_diverging: bool


class SectorPeerResponse(BaseModel):
    ticker: str
    sector: str
    industry: str
    rs_rating: int | None
    trend: str


class RelationshipMapResponse(BaseModel):
    ticker: str
    region: str
    statistical: list[StatisticalRelationResponse]
    sector_peers: list[SectorPeerResponse]
    computed_at: datetime
