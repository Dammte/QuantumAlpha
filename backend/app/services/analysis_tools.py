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


def historical_analogs(
    closes: pd.Series,
    lookback: int = 21,
    forward_horizon: int = 21,
    n_neighbors: int = 15,
    min_gap: int = 30,
) -> HistoricalAnalogs | None:
    """Finds the `n_neighbors` historical windows (in this ticker's own price
    history) whose momentum + volatility most resemble right now, and reports the
    distribution of what happened over the following `forward_horizon` days after
    those analogs. `min_gap` excludes the recent tail so "similar to last week"
    doesn't just match last week itself.
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

    last_valid = n - forward_horizon - 1 - min_gap
    start_idx = max(lookback, 20)
    candidates = [
        (i, trailing_return.iloc[i], volatility.iloc[i])
        for i in range(start_idx, last_valid)
        if not pd.isna(trailing_return.iloc[i]) and not pd.isna(volatility.iloc[i])
    ]
    if len(candidates) < n_neighbors:
        return None

    features = np.array([[c[1], c[2]] for c in candidates])
    mean, std = features.mean(axis=0), features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    current_normalized = (np.array([current_return, current_vol]) - mean) / std

    distances = np.linalg.norm(normalized - current_normalized, axis=1)
    nearest = np.argsort(distances)[:n_neighbors]

    forward_returns = np.array(
        [closes.iloc[candidates[j][0] + forward_horizon] / closes.iloc[candidates[j][0]] - 1 for j in nearest]
    )

    return HistoricalAnalogs(
        n_analogs=len(forward_returns),
        forward_horizon_days=forward_horizon,
        avg_forward_return=float(forward_returns.mean()),
        median_forward_return=float(np.median(forward_returns)),
        win_rate=float((forward_returns > 0).mean()),
    )
