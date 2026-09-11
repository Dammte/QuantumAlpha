from dataclasses import dataclass
from datetime import date

from app.domain.models.ticker_info import NewsArticle, TickerInfo
from app.services.analysis_tools import HistoricalAnalogs, MonthSeasonality
from app.services.backtest_engine import TripleBarrierBacktestResult
from app.services.multi_timeframe import MultiTimeframeRead
from app.services.recommendation_engine import Recommendation
from app.services.technical_analysis import ImminentCross, PriceLevel, Stage, TrendState


@dataclass(frozen=True, slots=True)
class PricePoint:
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


@dataclass(frozen=True, slots=True)
class TickerAnalysis:
    ticker: str
    name: str | None
    sector: str | None
    industry: str | None
    currency: str | None
    market_cap: float | None
    # Tercera auditoría, Bloque F-9: a per-ticker deep-dive field (like
    # market_cap above) - never computed for the whole universe screener,
    # only here and in premium_watchlist_service.py's bounded candidate
    # loop, to avoid a per-ticker network call in a hot path.
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
    trend: TrendState
    stage: Stage | None
    ma_cross: str | None
    imminent_cross: ImminentCross | None
    imminent_cross_short_term: ImminentCross | None
    candlestick_pattern: str | None
    mansfield_rs: float | None
    rs_rating: int | None
    minervini_score: int
    minervini_pass: bool
    support_resistance: list[PriceLevel]
    obv_divergence: str | None
    market_trend: TrendState | None  # informational only - see recommendation_engine.py docstring
    vix_regime: str | None  # informational only - see recommendation_engine.py docstring
    is_intraday_snapshot: bool
    multi_timeframe: MultiTimeframeRead
    confirmed_recommendation: Recommendation | None
    price_history: list[PricePoint]
    news: list[NewsArticle]
    fundamentals: TickerInfo | None
    seasonality: list[MonthSeasonality]
    historical_analogs: HistoricalAnalogs | None
    recommendation: Recommendation
    triple_barrier_backtest: TripleBarrierBacktestResult | None
