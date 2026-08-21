from dataclasses import dataclass

from app.services.technical_analysis import ImminentCross, Stage, TrendState


@dataclass(frozen=True, slots=True)
class TickerSnapshot:
    ticker: str
    sector: str
    industry: str | None
    cap_tier: str
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
    trend: TrendState
    stage: Stage | None
    ma_cross: str | None  # "golden" | "death" | None (SMA50 vs SMA200)
    minervini_score: int
    minervini_pass: bool
    rs_rating: int | None = None  # filled in later - needs the whole universe to percentile-rank
    currency: str = "USD"  # yfinance's own currency code; GBp means pence, not pounds (LSE convention)
    # Segunda auditoría, Bloque 3: inputs for the cross-sectional, setup-specific
    # percentile score (watchlist_service.setup_percentile_score) that replaces
    # RS Rating (12-month momentum) as the short-term tiers' ordering criterion -
    # RS Rating stays the ordering criterion for the monthly tier only, where a
    # 12-month momentum read is actually the right question to ask.
    atr_ratio_50d: float | None = None  # current ATR(14) vs its own 50-day average - <1 contracting, >1 expanding
    atr_multiple_sma21: float | None = None  # atr_multiple_from_sma(sma_window=21) - vs the 50-day one above
    range_position_20d: float | None = None  # rolling_position_in_range(close, 20) - 0 (low) to 1 (high)
    mansfield_rs_4w: float | None = None  # mansfield_rs(window=20, ~4 weeks) - vs the 200-day one above
    relative_volume_trend: float | None = None  # today's relative_volume minus its value 5 sessions ago
    # "Tendencia" screener, corto/mediano plazo (ago 2026): ma_cross (above) is
    # SMA50/SMA200 - a months-scale signal, too slow for a portfolio managed on
    # días/semanas (CLAUDE.md). ma_cross_short mirrors it on the same
    # FAST_MA_PERIOD/50 pair multi_timeframe.py/exit_engine.py/
    # ticker_analysis_service.py already use for exactly this reason - weeks,
    # not months. imminent_cross_short_term reuses detect_imminent_cross (same
    # function already wired into the single-ticker analysis view) to project
    # a crossover *before* it happens, not just report one that already did.
    ma_cross_short: str | None = None  # "golden" | "death" | None (SMA{FAST_MA_PERIOD} vs SMA50)
    imminent_cross_short_term: ImminentCross | None = None


@dataclass(frozen=True, slots=True)
class SectorPerformance:
    sector: str
    etf: str
    change_1d: float | None
    change_1w: float | None
    change_1m: float | None
    change_3m: float | None
    change_6m: float | None
    change_1y: float | None
    rs_rank: int | None = None  # 1-99 percentile among the 11 sectors, IBD RS-Rating style


@dataclass(frozen=True, slots=True)
class SectorForecast:
    """A *forward-looking* read on a sector, distinct from `SectorPerformance`
    (which only describes how it already did) - see `sector_forecast.py`."""

    sector: str
    etf: str
    current_state_label: str  # human label for the sector's current volatility-normalized state
    forecast_5d_return: float
    forecast_21d_return: float
    prob_bullish_21d: float
    has_statistical_structure: bool  # False -> this sector's own history looks like noise, treat with skepticism
    top_stocks: list[str]  # this region's universe's highest-RS-Rating names currently in this sector


@dataclass(frozen=True, slots=True)
class IndustryPerformance:
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
    leaders: list[TickerSnapshot]
