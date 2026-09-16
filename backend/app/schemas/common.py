"""Response models with no dependency on any other schema module - kept here
(rather than in `market.py`) purely so `quant_analysis.py` and `market.py` can
both use `PriceLevelResponse` without importing from each other."""

from pydantic import BaseModel


class PriceLevelResponse(BaseModel):
    price: float
    kind: str
    strength: int
    distance_pct: float


class LevelResponse(BaseModel):
    """See `technical_analysis.Level` (Parte 5.1) - the richer level engine
    alongside `PriceLevelResponse` above, not replacing it. `distance_atr`
    (not `distance_pct` alone) is the metric comparable across tickers of
    very different volatility; `state` captures a transition
    (breaking/broken/lost), not just current distance."""

    kind: str  # LevelKind - "ema21" | "ema55" | "sma50" | "sma200" | "weekly_ma30" | "pivot_resistance" | "pivot_support" | "range_high_20" | "range_low_20" | "high_52w" | "prior_day_high" | "prior_day_low"  # noqa: E501
    price: float
    side: str  # "above" | "below"
    distance_pct: float
    distance_atr: float
    state: str  # LevelState - "far"|"approaching"|"testing"|"breaking"|"broken_confirmed"|"lost_confirmed"
    bars_in_state: int
    strength: int | None
    slope_pct_20d: float | None
