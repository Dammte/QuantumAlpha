"""Extra tools for the single-ticker deep-dive analysis: a simplified Gann fan,
calendar seasonality, and a small-scale "historical analog" engine.

The historical-analog piece is a deliberately modest nod to the kind of pattern
matching quant funds like Renaissance are known for - at nowhere near that scale
or sophistication, but the same core idea: describe the current technical state
as a feature vector, find the most similar states in this same ticker's own
history, and report what tended to happen next. It's a k-nearest-neighbors
lookup over two features (momentum, volatility), not a trained model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.technical_analysis import TrendState


@dataclass(frozen=True, slots=True)
class GannLine:
    label: str
    values: list[float | None]  # aligned 1:1 with the input series' index


GANN_RATIOS: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1))


def find_gann_pivot(
    high: pd.Series, low: pd.Series, trend: TrendState, lookback: int = 120
) -> tuple[int, float] | None:
    """Anchor point for the fan: the lowest swing low in the trailing `lookback`
    bars for an uptrend (fan drawn upward from a base), or the highest swing high
    for a downtrend (fan drawn downward from a top)."""
    if high.empty or low.empty:
        return None
    window_high = high.iloc[-lookback:] if len(high) > lookback else high
    window_low = low.iloc[-lookback:] if len(low) > lookback else low

    if trend == TrendState.DOWNTREND:
        pos = high.index.get_loc(window_high.idxmax())
        return pos, float(window_high.max())
    pos = low.index.get_loc(window_low.idxmin())
    return pos, float(window_low.min())


def gann_fan(
    high: pd.Series,
    low: pd.Series,
    trend: TrendState,
    atr_series: pd.Series,
    lookback: int = 120,
) -> list[GannLine] | None:
    """A practical approximation of a Gann fan: real Gann angles assume a fixed,
    calibrated price-per-bar chart scale, which is subjective. Here the "1x1" unit
    is the ATR at the pivot bar (a standard, defensible stand-in for "typical price
    movement per bar"), and the fan angles (1x1, 2x1, 1x2) are projected forward
    from the pivot at multiples of that unit per bar.
    """
    pivot = find_gann_pivot(high, low, trend, lookback)
    if pivot is None:
        return None
    pivot_idx, pivot_price = pivot

    valid_atr = atr_series.dropna()
    if valid_atr.empty:
        return None
    unit = atr_series.iloc[pivot_idx]
    unit = float(unit) if not pd.isna(unit) else float(valid_atr.iloc[0])
    if unit == 0:
        return None

    direction = -1 if trend == TrendState.DOWNTREND else 1
    n = len(high)
    lines = []
    for num, den in GANN_RATIOS:
        slope = direction * unit * (num / den)
        values = [None if i < pivot_idx else pivot_price + slope * (i - pivot_idx) for i in range(n)]
        lines.append(GannLine(label=f"{num}x{den}", values=values))
    return lines


@dataclass(frozen=True, slots=True)
class MonthSeasonality:
    month: int  # 1-12
    avg_return: float
    win_rate: float
    n_observations: int


def seasonality_by_month(closes: pd.Series) -> list[MonthSeasonality]:
    """Historical average return and win rate for each calendar month, compounding
    daily closes to month-end first so a single volatile day doesn't get the same
    weight as a full month."""
    monthly_closes = closes.resample("ME").last()
    monthly_returns = monthly_closes.pct_change().dropna()

    results = []
    for month in range(1, 13):
        month_returns = monthly_returns[monthly_returns.index.month == month]
        if month_returns.empty:
            results.append(MonthSeasonality(month=month, avg_return=0.0, win_rate=0.0, n_observations=0))
        else:
            results.append(
                MonthSeasonality(
                    month=month,
                    avg_return=float(month_returns.mean()),
                    win_rate=float((month_returns > 0).mean()),
                    n_observations=int(len(month_returns)),
                )
            )
    return results


@dataclass(frozen=True, slots=True)
class HistoricalAnalogs:
    n_analogs: int
    forward_horizon_days: int
    avg_forward_return: float
    median_forward_return: float
    win_rate: float
    # Regime context (None when no VIX history was supplied - see
    # `regime_matched` below): what the broader market's fear/uncertainty
    # level actually was at each matched analog, not just this ticker's own
    # price shape. The closest quantifiable proxy available for "was there a
    # war/crisis/euphoria at the time" - see module docstring.
    regime_matched: bool
    current_vix_level: float | None
    avg_analog_vix_level: float | None
    pct_analogs_in_elevated_fear: float | None


VIX_ELEVATED_FEAR_LEVEL = 20.0  # matches market_context_service.vix_regime's "miedo elevado" threshold


def historical_analogs(
    closes: pd.Series,
    lookback: int = 21,
    forward_horizon: int = 21,
    n_neighbors: int = 15,
    min_gap: int = 30,
    vix_close: pd.Series | None = None,
) -> HistoricalAnalogs | None:
    """Finds the `n_neighbors` historical windows (in this ticker's own price
    history) whose momentum + volatility most resemble right now, and reports the
    distribution of what happened over the following `forward_horizon` days after
    those analogs. `min_gap` excludes the recent tail so "similar to last week"
    doesn't just match last week itself.

    When `vix_close` is supplied (the VIX index's own close series, any date
    range - it gets aligned to `closes`' index), the match also weighs a third
    dimension: what the broader market's fear/volatility level actually was at
    each candidate point, not just this ticker's own momentum/vol shape. Two
    stocks can have identical price-action fingerprints while one happened
    during a market-wide panic and the other during calm, untroubled markets -
    genuinely different contexts that a price-only match can't distinguish.
    True news-based context (wars, specific macro events) isn't available from
    this data source, so VIX level is the closest quantifiable, always-on
    proxy for "how fearful/uncertain was the market" at each historical point -
    the same proxy `MarketContextService` already uses for the *current*
    moment, applied retroactively here. Falls back to the plain 2-feature
    (momentum, volatility) match when `vix_close` isn't supplied.
    """
    n = len(closes)
    if n < lookback + forward_horizon + 252:
        return None

    returns = closes.pct_change()
    trailing_return = closes.pct_change(lookback)
    volatility = returns.rolling(20).std()

    current_return = trailing_return.iloc[-1]
    current_vol = volatility.iloc[-1]
    if pd.isna(current_return) or pd.isna(current_vol):
        return None

    # Gated on the *current* point having a real VIX reading, not just on
    # vix_close being non-empty: a series that doesn't actually overlap
    # `closes`' date range (reindex -> all-NaN) must fall back to the plain
    # 2-feature match rather than filter every single candidate out for
    # lacking a VIX value it was never going to have.
    aligned_vix: pd.Series | None = None
    current_vix_level: float | None = None
    if vix_close is not None and not vix_close.empty:
        candidate_vix = vix_close.reindex(closes.index).ffill()
        if not pd.isna(candidate_vix.iloc[-1]):
            aligned_vix = candidate_vix
            current_vix_level = float(candidate_vix.iloc[-1])
    regime_matched = aligned_vix is not None

    last_valid = n - forward_horizon - 1 - min_gap
    start_idx = max(lookback, 20)
    candidates = []
    for i in range(start_idx, last_valid):
        ret, vol = trailing_return.iloc[i], volatility.iloc[i]
        if pd.isna(ret) or pd.isna(vol):
            continue
        vix_at_i = aligned_vix.iloc[i] if aligned_vix is not None else None
        if aligned_vix is not None and pd.isna(vix_at_i):
            continue
        candidates.append((i, ret, vol, vix_at_i))
    if len(candidates) < n_neighbors:
        return None

    if regime_matched:
        features = np.array([[c[1], c[2], c[3]] for c in candidates])
        current_point = np.array([current_return, current_vol, current_vix_level])
    else:
        features = np.array([[c[1], c[2]] for c in candidates])
        current_point = np.array([current_return, current_vol])

    mean, std = features.mean(axis=0), features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    current_normalized = (current_point - mean) / std

    distances = np.linalg.norm(normalized - current_normalized, axis=1)
    nearest = np.argsort(distances)[:n_neighbors]

    forward_returns = np.array(
        [closes.iloc[candidates[j][0] + forward_horizon] / closes.iloc[candidates[j][0]] - 1 for j in nearest]
    )

    avg_analog_vix_level = None
    pct_analogs_in_elevated_fear = None
    if regime_matched:
        analog_vix_levels = np.array([candidates[j][3] for j in nearest])
        avg_analog_vix_level = float(analog_vix_levels.mean())
        pct_analogs_in_elevated_fear = float((analog_vix_levels >= VIX_ELEVATED_FEAR_LEVEL).mean())

    return HistoricalAnalogs(
        n_analogs=len(forward_returns),
        forward_horizon_days=forward_horizon,
        avg_forward_return=float(forward_returns.mean()),
        median_forward_return=float(np.median(forward_returns)),
        win_rate=float((forward_returns > 0).mean()),
        regime_matched=regime_matched,
        current_vix_level=current_vix_level,
        avg_analog_vix_level=avg_analog_vix_level,
        pct_analogs_in_elevated_fear=pct_analogs_in_elevated_fear,
    )
