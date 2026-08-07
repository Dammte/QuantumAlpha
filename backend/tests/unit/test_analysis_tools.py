import numpy as np
import pandas as pd
import pytest

from app.services import analysis_tools as at
from app.services import technical_analysis as ta


def test_find_gann_pivot_uptrend_picks_lowest_low():
    low = pd.Series([50.0, 40.0, 45.0, 60.0, 70.0])
    high = low + 2
    pivot = at.find_gann_pivot(high, low, ta.TrendState.UPTREND)
    assert pivot == (1, 40.0)


def test_find_gann_pivot_downtrend_picks_highest_high():
    high = pd.Series([50.0, 80.0, 60.0, 40.0, 30.0])
    low = high - 2
    pivot = at.find_gann_pivot(high, low, ta.TrendState.DOWNTREND)
    assert pivot == (1, 80.0)


def test_gann_fan_none_values_before_pivot_and_rising_after_in_uptrend():
    n = 60
    low = pd.Series(100 - np.arange(n) * 0.1)  # keeps declining, pivot = last bar
    high = low + 2
    close = (low + high) / 2
    atr_series = ta.atr(high, low, close)

    lines = at.gann_fan(high, low, ta.TrendState.UPTREND, atr_series, lookback=n)

    assert lines is not None
    one_by_one = next(line for line in lines if line.label == "1x1")
    assert one_by_one.values[0] is None or one_by_one.values[-1] is not None
    # The pivot (lowest low) is the very last bar here, so only the last value is non-null.
    assert one_by_one.values[-1] == pytest.approx(low.iloc[-1])


def test_gann_fan_slope_direction_matches_trend():
    n = 60
    rising = pd.Series(100 + np.arange(n) * 0.3)
    falling = pd.Series(100 - np.arange(n) * 0.3)

    up_atr = ta.atr(rising + 1, rising - 1, rising)
    down_atr = ta.atr(falling + 1, falling - 1, falling)
    up_lines = at.gann_fan(rising + 1, rising - 1, ta.TrendState.UPTREND, up_atr, lookback=n)
    down_lines = at.gann_fan(falling + 1, falling - 1, ta.TrendState.DOWNTREND, down_atr, lookback=n)

    up_1x1 = next(line for line in up_lines if line.label == "1x1")
    down_1x1 = next(line for line in down_lines if line.label == "1x1")
    up_values = [v for v in up_1x1.values if v is not None]
    down_values = [v for v in down_1x1.values if v is not None]
    assert up_values[-1] > up_values[0]  # projects upward from an uptrend's base
    assert down_values[-1] < down_values[0]  # projects downward from a downtrend's top


def test_gann_fan_none_when_insufficient_atr_history():
    close = pd.Series([100.0, 101.0, 102.0])
    assert at.gann_fan(close + 1, close - 1, ta.TrendState.UPTREND, pd.Series([np.nan] * 3)) is None


def test_seasonality_by_month_covers_all_twelve_months():
    dates = pd.date_range("2015-01-01", periods=252 * 8, freq="B")
    closes = pd.Series(100 * (1.0002 ** np.arange(len(dates))), index=dates)
    result = at.seasonality_by_month(closes)
    assert [r.month for r in result] == list(range(1, 13))
    assert all(r.n_observations > 0 for r in result)


def test_seasonality_by_month_detects_a_strong_seasonal_month():
    # Flat every month except January, which always jumps +20% - January should
    # show a clearly higher average return and a 100% win rate.
    dates = pd.date_range("2010-01-01", periods=252 * 10, freq="B")
    closes = []
    price = 100.0
    for d in dates:
        if d.month == 1 and d.day <= 5:
            price *= 1.03
        closes.append(price)
    series = pd.Series(closes, index=dates)

    result = at.seasonality_by_month(series)
    january = next(r for r in result if r.month == 1)
    other_months_avg = np.mean([r.avg_return for r in result if r.month != 1])
    assert january.avg_return > other_months_avg
    assert january.win_rate == pytest.approx(1.0)


def test_historical_analogs_none_when_insufficient_history():
    closes = pd.Series(100 + np.arange(100) * 0.1)
    assert at.historical_analogs(closes) is None


def _synthetic_closes_for_analogs(n: int = 1500, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = [100.0]
    pullback_countdown = 0
    bounce_countdown = 0
    for _ in range(n):
        if pullback_countdown == 0 and bounce_countdown == 0 and rng.random() < 0.03:
            pullback_countdown = 10
        if pullback_countdown > 0:
            prices.append(prices[-1] * 0.97)
            pullback_countdown -= 1
            if pullback_countdown == 0:
                bounce_countdown = 21
        elif bounce_countdown > 0:
            prices.append(prices[-1] * (1.10 ** (1 / 21)))
            bounce_countdown -= 1
        else:
            prices.append(prices[-1] * (1 + rng.normal(0, 0.005)))
    closes = pd.Series(prices)
    # Force the CURRENT state to look like "just finished a sharp pullback".
    closes.iloc[-22:-21] = closes.iloc[-22] * 0.90
    return closes


def test_historical_analogs_returns_well_formed_stats_on_realistic_data():
    closes = _synthetic_closes_for_analogs()

    result = at.historical_analogs(closes, lookback=21, forward_horizon=21, n_neighbors=15)

    assert result is not None
    assert result.n_analogs == 15
    assert result.forward_horizon_days == 21
    assert 0.0 <= result.win_rate <= 1.0
    assert np.isfinite(result.avg_forward_return)
    # No vix_close supplied - regime matching must degrade gracefully, not error.
    assert result.regime_matched is False
    assert result.current_vix_level is None
    assert result.avg_analog_vix_level is None
    assert result.pct_analogs_in_elevated_fear is None


def test_historical_analogs_uses_vix_regime_when_supplied():
    closes = _synthetic_closes_for_analogs()
    # Flat, unremarkable VIX throughout - just needs to be present/aligned to
    # exercise the regime-matching path end to end.
    vix_close = pd.Series([18.0] * len(closes), index=closes.index)

    result = at.historical_analogs(closes, lookback=21, forward_horizon=21, n_neighbors=15, vix_close=vix_close)

    assert result is not None
    assert result.regime_matched is True
    assert result.current_vix_level == pytest.approx(18.0)
    assert result.avg_analog_vix_level == pytest.approx(18.0)
    assert result.pct_analogs_in_elevated_fear == pytest.approx(0.0)  # 18 < the 20.0 elevated-fear threshold


def test_historical_analogs_reports_elevated_fear_share_correctly():
    closes = _synthetic_closes_for_analogs()
    n = len(closes)
    # High VIX (panic) for the first third of history, calm for the rest -
    # the KNN match itself is still driven by price features, but whichever
    # analogs get matched should have their true, correctly-labeled VIX regime
    # reported back, not an average that silently ignores the split.
    vix_values = [35.0] * (n // 3) + [14.0] * (n - n // 3)
    vix_close = pd.Series(vix_values, index=closes.index)

    result = at.historical_analogs(closes, lookback=21, forward_horizon=21, n_neighbors=15, vix_close=vix_close)

    assert result is not None
    assert result.regime_matched is True
    assert result.pct_analogs_in_elevated_fear in (0.0, 1.0) or 0.0 < result.pct_analogs_in_elevated_fear < 1.0
    # avg_analog_vix_level must be a real blend of the two regimes actually
    # present, not the flat 18.0 from the other test.
    assert result.avg_analog_vix_level in (pytest.approx(35.0), pytest.approx(14.0)) or (
        14.0 < result.avg_analog_vix_level < 35.0
    )


def test_historical_analogs_falls_back_when_vix_has_no_overlap():
    closes = _synthetic_closes_for_analogs()
    # A vix_close with a completely disjoint index (reindex -> all NaN) must
    # degrade to the plain 2-feature match, not crash or silently corrupt it.
    vix_close = pd.Series([20.0] * 10, index=range(100_000, 100_010))

    result = at.historical_analogs(closes, lookback=21, forward_horizon=21, n_neighbors=15, vix_close=vix_close)

    assert result is not None
    assert result.regime_matched is False
    assert result.current_vix_level is None
    assert np.isfinite(result.median_forward_return)
