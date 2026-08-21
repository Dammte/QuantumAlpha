from datetime import date, datetime

from pydantic import BaseModel

from app.schemas.common import PriceLevelResponse
from app.schemas.quant_analysis import (
    EntryTimingResponse,
    GarchResponse,
    ImminentCrossResponse,
    KellyPositionSizeResponse,
    MarkovChainResponse,
    MonteCarloResponse,
    MultiTimeframeResponse,
    RecommendationResponse,
    StatisticalStructureResponse,
    TripleBarrierBacktestResponse,
    WalkForwardBacktestResponse,
)


class PricePointResponse(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma20: float | None
    sma50: float | None
    sma150: float | None
    sma200: float | None
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    gann_1x1: float | None
    gann_1x2: float | None
    gann_2x1: float | None
    rsi14: float | None
    macd_histogram: float | None


class NewsArticleResponse(BaseModel):
    title: str
    publisher: str | None
    link: str | None
    published_at: str | None


class InstitutionalHolderResponse(BaseModel):
    holder: str
    shares: float | None
    value: float | None
    pct_held: float | None
    date_reported: str | None


class HoldersSummaryResponse(BaseModel):
    pct_held_by_institutions: float | None
    pct_held_by_insiders: float | None
    top_institutional_holders: list[InstitutionalHolderResponse]


class FundamentalsResponse(BaseModel):
    name: str | None
    sector: str | None
    industry: str | None
    currency: str | None
    market_cap: float | None
    trailing_pe: float | None
    forward_pe: float | None
    dividend_yield: float | None
    beta: float | None
    average_volume: float | None
    analyst_recommendation: str | None
    analyst_target_mean_price: float | None
    analyst_opinion_count: int | None
    revenue_growth: float | None
    profit_margins: float | None
    debt_to_equity: float | None


class MonthSeasonalityResponse(BaseModel):
    month: int
    avg_return: float
    win_rate: float
    n_observations: int


class HistoricalAnalogsResponse(BaseModel):
    n_analogs: int
    forward_horizon_days: int
    avg_forward_return: float
    median_forward_return: float
    win_rate: float
    regime_matched: bool
    current_vix_level: float | None
    avg_analog_vix_level: float | None
    pct_analogs_in_elevated_fear: float | None


class RecommendationSnapshotFactorResponse(BaseModel):
    label: str
    points: int
    triggered: bool


class RecommendationSnapshotResponse(BaseModel):
    id: int
    ticker: str
    created_at: datetime
    verdict: str
    score: int
    price: float
    currency: str
    horizon: str
    engine_version: str
    factors: list[RecommendationSnapshotFactorResponse]


class TickerAnalysisResponse(BaseModel):
    ticker: str
    name: str | None
    sector: str | None
    industry: str | None
    currency: str | None
    market_cap: float | None
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
    macd_line: float | None
    macd_signal: float | None
    macd_histogram: float | None
    adx14: float | None
    plus_di: float | None
    minus_di: float | None
    atr14: float | None
    atr_multiple: float | None
    sma20: float | None
    sma50: float | None
    sma150: float | None
    sma200: float | None
    dist_52w_high: float | None
    dist_52w_low: float | None
    trend: str
    stage: str | None
    ma_cross: str | None
    imminent_cross: ImminentCrossResponse | None
    imminent_cross_short_term: ImminentCrossResponse | None
    candlestick_pattern: str | None
    mansfield_rs: float | None
    rs_rating: int | None
    minervini_score: int
    minervini_pass: bool
    support_resistance: list[PriceLevelResponse]
    obv_divergence: str | None
    statistical_structure: StatisticalStructureResponse | None
    market_trend: str | None
    vix_regime: str | None
    is_intraday_snapshot: bool
    multi_timeframe: MultiTimeframeResponse
    confirmed_recommendation: RecommendationResponse | None
    price_history: list[PricePointResponse]
    news: list[NewsArticleResponse]
    fundamentals: FundamentalsResponse | None
    holders: HoldersSummaryResponse | None
    seasonality: list[MonthSeasonalityResponse]
    historical_analogs: HistoricalAnalogsResponse | None
    recommendation: RecommendationResponse
    entry_timing: EntryTimingResponse | None
    markov: MarkovChainResponse | None
    garch: GarchResponse | None
    monte_carlo: MonteCarloResponse | None
    backtest: WalkForwardBacktestResponse | None
    triple_barrier_backtest: TripleBarrierBacktestResponse | None
    position_sizing: KellyPositionSizeResponse | None
