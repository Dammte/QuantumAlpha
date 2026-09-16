from dataclasses import dataclass
from datetime import date

from app.domain.models.ticker_info import NewsArticle, TickerInfo
from app.services.backtest_engine import TripleBarrierBacktestResult
from app.services.levels_engine import GateResult, GradeResult
from app.services.multi_timeframe import MultiTimeframeRead
from app.services.technical_analysis import ImminentCross, Level, PriceLevel, Stage, TrendState


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
    # Parte 3.2/11: the real EMA21/EMA55 pair the gate/exit engine actually
    # decide against - drawn on the chart alongside the SMAs above, a
    # confirmed gap before this (only SMA50/SMA200 were ever visible, never
    # the pair the whole system depends on). Same mtf.FAST_MA_PERIOD/
    # SLOW_MA_PERIOD basis as everywhere else, not a fifth definition.
    ema21: float | None
    ema55: float | None
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
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
    # to avoid a per-ticker network call in a hot path.
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
    levels: list[Level]  # Parte 5.1 - ver CoreTickerSignals.levels
    obv_divergence: str | None
    market_trend: TrendState | None  # informational only - see recommendation_engine.py docstring
    vix_regime: str | None  # informational only - see recommendation_engine.py docstring
    is_intraday_snapshot: bool
    multi_timeframe: MultiTimeframeRead
    confirmed_gate: GateResult | None
    price_history: list[PricePoint]
    news: list[NewsArticle]
    fundamentals: TickerInfo | None
    gate: GateResult
    grade: GradeResult | None  # Parte 5.3 - ver CoreTickerSignals.grade
    triple_barrier_backtest: TripleBarrierBacktestResult | None
    # Reconstruction (2026-09), Fase 7: `None` whenever the Gemini narrator
    # isn't configured (the default) or its call failed for any reason - see
    # `app.domain.interfaces.llm_narrator.LLMNarrator`. Never affects `gate`
    # above, which is always computed first and handed to the narrator as an
    # already-settled fact.
    llm_narrative: str | None
