from datetime import date, datetime

from pydantic import BaseModel

from app.schemas.common import PriceLevelResponse
from app.schemas.quant_analysis import CoreSignalsResponse, MultiTimeframeResponse


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


class SectorPerformanceResponse(BaseModel):
    sector: str
    etf: str
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_6m: float | None
    change_1y: float | None
    rs_rank: int | None


class SectorForecastResponse(BaseModel):
    """Forward-looking counterpart to `SectorPerformanceResponse` - see
    `MarketScreenerService.get_sector_forecast`."""

    sector: str
    etf: str
    current_state_label: str
    forecast_5d_return: float
    forecast_21d_return: float
    prob_bullish_21d: float
    has_statistical_structure: bool
    top_stocks: list[str]


class SectorRotationResponse(BaseModel):
    leaders: list[str]
    laggards: list[str]
    cycle_phase: str | None
    cycle_confidence: float
    cycle_description: str | None
    defensive_leadership: bool
    warning: str | None


class IndustryPerformanceResponse(BaseModel):
    industry: str
    sector: str
    etf: str | None
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_6m: float | None
    change_1y: float | None
    avg_rs_rating: float | None
    leaders: list[TickerSnapshotResponse]


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


class FearGreedResponse(BaseModel):
    score: float
    label: str
    components: dict[str, float]


class LiquidityResponse(BaseModel):
    proxy_ticker: str
    trend: str
    headwind: bool


class MarketRegimeResponse(BaseModel):
    verdict: str
    headline: str
    reasons: list[str]


class MacroSnapshotResponse(BaseModel):
    yield_curve_spread: float | None
    yield_curve_date: str | None
    yield_curve_inverted: bool
    unemployment_rate: float | None
    unemployment_date: str | None
    cpi_yoy_change: float | None
    cpi_date: str | None


class MarketContextResponse(BaseModel):
    indices: list[IndexSnapshotResponse]
    vix: VixSnapshotResponse
    fear_greed: FearGreedResponse
    liquidity: LiquidityResponse
    regime: MarketRegimeResponse
    news: list[NewsArticleResponse]
    macro: MacroSnapshotResponse | None


class WatchlistItemResponse(BaseModel):
    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    horizon: str
    reasons: list[str]
    snapshot: TickerSnapshotResponse
    sector_rs_rank: int | None
    # "oversold_bounce" | "breakout_volume" | "trend_continuation" |
    # "pullback_to_support" for a short-term item, `None` for medium/long-term
    # ones (not split into setup types) - see watchlist_service.py.
    setup: str | None = None
    percentile_score: float | None = None


class PremiumWatchlistItemResponse(BaseModel):
    """A ticker that didn't just match a cheap technical rule, but was run through
    the full "Analizar activo" pipeline (recommendation, GARCH, Markov, Monte Carlo,
    walk-forward backtest, Kelly sizing) and came out endorsed - see
    `premium_watchlist_service.py` for the approval bar."""

    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
    currency: str
    region: str  # "us" | "europe"
    tier: str  # "daily" | "weekly" | "monthly"
    reasons: list[str]
    premium_score: float
    signals: CoreSignalsResponse
    setup: str | None = None


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


class SwapSuggestionResponse(BaseModel):
    """See opportunity_cost.py - a held position that isn't broken, but where a
    materially stronger, already-vetted premium candidate exists instead."""

    held_ticker: str
    held_score: int
    held_signal: str
    candidate_ticker: str
    candidate_score: float
    candidate_sector: str
    candidate_currency: str


class PortfolioRiskResponse(BaseModel):
    positions: list[PositionRiskResponse]
    swap_suggestions: list[SwapSuggestionResponse]
    computed_at: datetime  # when this was last actually computed - may be hours old, see durable_cache.py


class TierDiscardStatsResponse(BaseModel):
    """"15 de 47 candidatos analizados" - see premium_watchlist_service.TierDiscardStats."""

    tier: str
    prefilter_matches: int
    analyzed: int
    approved: int


class PremiumWatchlistResponse(BaseModel):
    items: list[PremiumWatchlistItemResponse]
    computed_at: datetime  # oldest of the tier(s) shown - see durable_cache.py
    discard_stats: list[TierDiscardStatsResponse] = []


class WatchlistResponse(BaseModel):
    items: list[WatchlistItemResponse]
    computed_at: datetime


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
