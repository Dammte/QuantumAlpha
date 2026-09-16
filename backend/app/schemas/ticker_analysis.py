from datetime import date, datetime

from pydantic import BaseModel

from app.schemas.common import LevelResponse, PriceLevelResponse
from app.schemas.quant_analysis import (
    GateResultResponse,
    GradeResponse,
    ImminentCrossResponse,
    MultiTimeframeResponse,
    TripleBarrierBacktestResponse,
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
    ema21: float | None
    ema55: float | None
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    rsi14: float | None
    macd_histogram: float | None


class NewsArticleResponse(BaseModel):
    title: str
    publisher: str | None
    link: str | None
    published_at: str | None


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
    revenue_growth: float | None
    profit_margins: float | None
    debt_to_equity: float | None


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
    days_to_earnings: int | None
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
    levels: list[LevelResponse]
    obv_divergence: str | None
    market_trend: str | None
    vix_regime: str | None
    is_intraday_snapshot: bool
    multi_timeframe: MultiTimeframeResponse
    confirmed_gate: GateResultResponse | None
    price_history: list[PricePointResponse]
    news: list[NewsArticleResponse]
    fundamentals: FundamentalsResponse | None
    gate: GateResultResponse
    grade: GradeResponse | None
    triple_barrier_backtest: TripleBarrierBacktestResponse | None
    # Reconstruction (2026-09), Fase 7: `None` unless GEMINI_API_KEY is set -
    # see LLMNarrator's own docstring. Purely explanatory text over `gate`
    # above, never a second verdict.
    llm_narrative: str | None
