from dataclasses import dataclass

from app.services.technical_analysis import Stage, TrendState


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
