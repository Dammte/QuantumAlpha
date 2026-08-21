from datetime import UTC, date, datetime, time

import numpy as np
import pandas as pd
import pytest

from app.services import technical_analysis as ta


def _ohlcv_df(dates, closes, volumes=None, wiggle=1.0):
    # The DatetimeIndex has to be attached to `closes` *before* deriving
    # open/high/low from it - building each column as a bare, unindexed
    # pd.Series and only assigning the DatetimeIndex on the final
    # pd.DataFrame(..., index=...) call silently reindex-aligns every column
    # to all-NaN (RangeIndex labels don't match DatetimeIndex labels).
    index = pd.to_datetime(list(dates))
    closes = pd.Series(closes, index=index, dtype=float)
    volumes = pd.Series(volumes, index=index, dtype=float) if volumes is not None else pd.Series(
        [1000.0] * len(closes), index=index
    )
    return pd.DataFrame(
        {
            "open": closes - wiggle / 2,
            "high": closes + wiggle,
            "low": closes - wiggle,
            "close": closes,
            "volume": volumes,
        }
    )


def test_resample_ohlcv_weekly_aggregates_ohlc_and_volume_correctly():
    # Two full business weeks: Mon 2024-01-01 - Fri 2024-01-05, then
    # Mon 2024-01-08 - Fri 2024-01-12. Both weeks end exactly on their
    # Friday, so both are "closed" and neither is dropped even without
    # include_partial.
    dates = pd.bdate_range("2024-01-01", periods=10)
    closes = [10, 11, 9, 12, 13, 20, 22, 18, 25, 24]
    volumes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    df = _ohlcv_df(dates, closes, volumes, wiggle=1.0)

    weekly = ta.resample_ohlcv(df, rule="W-FRI")

    assert len(weekly) == 2
    week1, week2 = weekly.iloc[0], weekly.iloc[1]
    assert week1["open"] == pytest.approx(10 - 0.5)  # first bar's open
    assert week1["high"] == pytest.approx(14.0)  # max(close)+wiggle over the week
    assert week1["low"] == pytest.approx(8.0)  # min(close)-wiggle over the week
    assert week1["close"] == pytest.approx(13.0)  # last bar's close
    assert week1["volume"] == pytest.approx(1500.0)  # sum
    assert week2["close"] == pytest.approx(24.0)
    assert week2["volume"] == pytest.approx(4000.0)
    assert weekly.index[0].date() == date(2024, 1, 5)  # labeled by the week's Friday
    assert weekly.index[1].date() == date(2024, 1, 12)


def test_resample_ohlcv_drops_still_forming_week_by_default():
    # Two complete weeks (as above) plus a single Monday of a third week -
    # that third week hasn't finished trading, so it must not be reported as
    # a "weekly bar" unless explicitly asked for. `now` pinned to that same
    # Monday - resample_ohlcv compares *today's* calendar date against the
    # period's own right edge, not the raw data's last bar date (see its
    # docstring on why that used to be wrong).
    dates = list(pd.bdate_range("2024-01-01", periods=10)) + [pd.Timestamp("2024-01-15")]
    closes = [10, 11, 9, 12, 13, 20, 22, 18, 25, 24, 30]
    df = _ohlcv_df(dates, closes)
    now = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)

    default = ta.resample_ohlcv(df, rule="W-FRI", now=now)
    assert len(default) == 2

    with_partial = ta.resample_ohlcv(df, rule="W-FRI", include_partial=True)
    assert len(with_partial) == 3
    assert with_partial.iloc[2]["close"] == pytest.approx(30.0)
    assert with_partial.iloc[2]["volume"] == pytest.approx(1000.0)  # only that one Monday


def test_resample_ohlcv_empty_df_returns_empty():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert ta.resample_ohlcv(empty).empty


def test_resample_ohlcv_keeps_a_complete_week_when_the_period_edge_was_a_calendar_holiday():
    # Segunda auditoría, Bloque 2: a Friday holiday (more common in Europe)
    # means the week's last raw bar falls on Thursday, short of the
    # resampled period's own calendar-Friday right edge - the old check
    # (the raw data's own last bar date vs. that edge) read this as "still
    # forming" and discarded an already-complete week. `now` pinned well
    # after that Friday: the week is unambiguously over regardless of
    # whether Friday itself had a bar.
    dates = pd.bdate_range("2024-01-01", periods=4)  # Mon-Thu; Fri 2024-01-05 is the simulated holiday
    df = _ohlcv_df(dates, [10, 11, 12, 13])
    now = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)

    weekly = ta.resample_ohlcv(df, rule="W-FRI", now=now)
    assert len(weekly) == 1
    assert weekly.iloc[0]["close"] == pytest.approx(13.0)


def test_resample_ohlcv_keeps_a_complete_month_ending_on_a_weekend():
    # 2024-03-31 (the "ME" period's calendar right edge) is a Sunday - the
    # last actual trading day is Friday 2024-03-29, short of that edge.
    # ~5/12 months end on a weekend, so `rule="ME"` hit this constantly.
    dates = pd.bdate_range("2024-03-25", periods=5)  # Mon 3/25 .. Fri 3/29
    df = _ohlcv_df(dates, [10, 11, 12, 13, 14])
    now = datetime(2024, 4, 5, 12, 0, tzinfo=UTC)

    monthly = ta.resample_ohlcv(df, rule="ME", now=now)
    assert len(monthly) == 1
    assert monthly.iloc[0]["close"] == pytest.approx(14.0)


def test_closed_bars_drops_the_bar_dated_today_before_the_close_cutoff():
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _ohlcv_df(dates, [10, 11, 12, 13, 14])
    # Same calendar day as the last bar, but still mid-session (10:00 UTC is
    # well before CLOSED_BAR_CUTOFF_UTC) - the bar could still move before
    # the actual close, so it's excluded exactly like the old date-only check.
    result = ta.closed_bars(df, now=datetime(2024, 1, 5, 10, 0, tzinfo=UTC))
    assert len(result) == 4
    assert result.index[-1] == dates[-2]


def test_closed_bars_keeps_todays_bar_once_past_the_close_cutoff():
    # The real-world gap this closes: markets had genuinely closed hours
    # earlier (e.g. 21:56 UTC, well past the US close), but the old
    # calendar-date-only check still excluded the day's already-final bar
    # until UTC midnight - a several-hour nightly blackout on every discrete
    # signal (crosses, price-vs-MA, patterns), which is exactly what let a
    # same-day breakdown go undetected until the next morning.
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _ohlcv_df(dates, [10, 11, 12, 13, 14])
    result = ta.closed_bars(df, now=datetime(2024, 1, 5, 22, 0, tzinfo=UTC))
    assert len(result) == 5
    assert result.index[-1] == dates[-1]


def test_closed_bars_right_at_the_cutoff_boundary_is_still_settled():
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _ohlcv_df(dates, [10, 11, 12, 13, 14])
    result = ta.closed_bars(df, now=datetime(2024, 1, 5, 21, 30, tzinfo=UTC))
    assert len(result) == 5


def test_closed_bars_keeps_every_bar_when_the_last_one_is_not_today():
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _ohlcv_df(dates, [10, 11, 12, 13, 14])
    result = ta.closed_bars(df, now=datetime(2024, 1, 8, 10, 0, tzinfo=UTC))
    assert len(result) == 5


def test_closed_bars_uses_a_custom_cutoff_instead_of_the_default():
    # Segunda auditoría, Bloque 2: a bar dated today at 17:00 UTC - past
    # Europe's real close (~16:30 UTC) but well before the US-centric
    # default (21:30 UTC). A caller that knows the ticker's region (see
    # market_universe.closed_bar_cutoff_for_ticker) can pass the right one.
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _ohlcv_df(dates, [10, 11, 12, 13, 14])
    now = datetime(2024, 1, 5, 17, 0, tzinfo=UTC)

    assert len(ta.closed_bars(df, now=now)) == 4  # default (US) cutoff: still "forming"
    assert len(ta.closed_bars(df, now=now, cutoff=time(16, 30))) == 5  # Europe: already settled


def test_closed_bars_empty_df_returns_empty():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert ta.closed_bars(empty).empty


def test_detect_cross_with_quality_strong_golden_cross():
    # slow is flat at 100; fast sits well below it, then crosses decisively
    # with growing separation, a positive slope, and a volume spike exactly
    # on the confirming bar.
    slow = pd.Series([100.0] * 30)
    fast = pd.Series([95.0] * 24 + [96.0, 98.0, 101.0, 104.0, 108.0, 112.0])
    close = fast
    high, low = close + 1, close - 1
    volume = pd.Series([1000.0] * 26 + [5000.0] + [1000.0] * 3)  # spike at index 26, the cross bar

    result = ta.detect_cross_with_quality(fast, slow, high, low, close, volume)

    assert result is not None
    assert result.direction == "golden"
    assert result.bars_since == 3  # confirmed at index 26, latest bar is index 29
    assert result.separation_atr is not None and result.separation_atr > ta.CROSS_STRONG_SEPARATION_ATR
    assert result.fast_slope > result.slow_slope  # diverging upward
    assert result.volume_confirmation is not None and result.volume_confirmation > ta.CROSS_STRONG_RELATIVE_VOLUME
    assert result.quality == "strong"


def test_detect_cross_with_quality_downgrades_to_weak_without_volume_confirmation():
    # Identical setup to the "strong" case above, minus the volume spike -
    # separation and slope alone aren't enough.
    slow = pd.Series([100.0] * 30)
    fast = pd.Series([95.0] * 24 + [96.0, 98.0, 101.0, 104.0, 108.0, 112.0])
    close = fast
    high, low = close + 1, close - 1
    volume = pd.Series([1000.0] * 30)  # flat - no confirmation on the cross bar

    result = ta.detect_cross_with_quality(fast, slow, high, low, close, volume)

    assert result is not None
    assert result.quality == "weak"


def test_detect_cross_with_quality_noise_when_separation_is_negligible():
    # fast and slow are practically overlapping - the sign technically flips,
    # but by a fraction of a single ATR, not a real separation.
    slow = pd.Series([100.0] * 30)
    fast = pd.Series([99.9] * 25 + [100.05] * 5)
    close = fast
    high, low = close + 1, close - 1

    result = ta.detect_cross_with_quality(fast, slow, high, low, close)

    assert result is not None
    assert result.direction == "golden"
    assert result.separation_atr is not None and result.separation_atr < ta.CROSS_NOISE_SEPARATION_ATR
    assert result.quality == "noise"


def test_detect_cross_with_quality_none_when_no_cross_happened():
    slow = pd.Series([100.0] * 30)
    fast = pd.Series([90.0] * 30)  # always below, never crosses
    assert ta.detect_cross_with_quality(fast, slow) is None


def test_detect_cross_with_quality_reads_the_cross_bars_volume_despite_real_warmup_nan():
    # Real SMA50/SMA200-style warmup, unlike every other test in this file
    # (which hand off already-full, NaN-free series): the first 199 bars are
    # NaN, matching this function's actual live inputs (`fast`/`slow` are
    # always `ta.sma(close, N)`, never raw data). `diff = (fast-slow).dropna()`
    # drops those 199 NaN rows, so diff's own internal position 0 sits at
    # calendar position 199 - exactly the offset a naive `.iloc` reuse into
    # `volume` (never truncated the same way) would silently miss, reading
    # volume from near the very start of the series instead of around the
    # actual cross.
    n_nan = 199
    dates = pd.bdate_range("2024-01-01", periods=n_nan + 8)
    nan_prefix = [np.nan] * n_nan
    slow = pd.Series(nan_prefix + [100.0] * 8, index=dates)
    # Crosses from below (-1) to above (+1) at calendar position n_nan + 4.
    fast = pd.Series(nan_prefix + [95.0, 96.0, 98.0, 99.0, 101.0, 104.0, 108.0, 112.0], index=dates)
    close = fast
    high, low = close + 1, close - 1
    volume = pd.Series([1000.0] * (n_nan + 8), index=dates)
    volume.iloc[n_nan + 4] = 5000.0  # spike exactly on the cross bar

    result = ta.detect_cross_with_quality(fast, slow, high, low, close, volume)

    assert result is not None
    assert result.direction == "golden"
    # Would read ~1.0 (no spike) under the old `.iloc[:cross_position + 1]`
    # bug, which lands near calendar position 4 - the NaN warmup region -
    # instead of the real cross bar at calendar position n_nan + 4.
    assert result.volume_confirmation is not None
    assert result.volume_confirmation > ta.CROSS_STRONG_RELATIVE_VOLUME


def test_detect_cross_with_quality_works_without_optional_series():
    # high/low/close/volume are all optional - separation_atr and
    # volume_confirmation degrade to None rather than raising.
    slow = pd.Series([100.0] * 10)
    fast = pd.Series([95.0] * 5 + [105.0] * 5)
    result = ta.detect_cross_with_quality(fast, slow)
    assert result is not None
    assert result.direction == "golden"
    assert result.separation_atr is None
    assert result.volume_confirmation is None


def test_consecutive_closes_below_counts_back_from_the_latest_bar():
    closes = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0])
    level = pd.Series([98.5] * 5)  # last 3 closes (98, 97, 96) are below 98.5
    assert ta.consecutive_closes_below(closes, level) == 3


def test_consecutive_closes_below_stops_at_the_first_bar_at_or_above():
    closes = pd.Series([90.0, 105.0, 96.0, 97.0])
    level = pd.Series([100.0] * 4)
    # latest two (96, 97) are below; the 105 two bars back breaks the streak
    assert ta.consecutive_closes_below(closes, level) == 2


def test_consecutive_closes_below_zero_when_latest_close_is_above():
    closes = pd.Series([90.0, 91.0, 105.0])
    level = pd.Series([100.0] * 3)
    assert ta.consecutive_closes_below(closes, level) == 0


def test_sma_basic():
    series = pd.Series([1, 2, 3, 4, 5])
    result = ta.sma(series, 3)
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[-1] == pytest.approx(4.0)
    assert pd.isna(result.iloc[0])


def test_bollinger_bands_upper_above_middle_above_lower():
    closes = pd.Series(100 + np.sin(np.arange(60) / 3) * 5)
    middle, upper, lower = ta.bollinger_bands(closes, window=20)
    valid = middle.dropna().index
    assert (upper[valid] >= middle[valid]).all()
    assert (middle[valid] >= lower[valid]).all()


def test_bollinger_bands_widen_with_more_volatility():
    calm = pd.Series([100.0] * 30)
    volatile = pd.Series(100 + np.array([((-1) ** i) * 5 for i in range(30)]))
    _, calm_upper, calm_lower = ta.bollinger_bands(calm, window=20)
    _, vol_upper, vol_lower = ta.bollinger_bands(volatile, window=20)
    assert (vol_upper.iloc[-1] - vol_lower.iloc[-1]) > (calm_upper.iloc[-1] - calm_lower.iloc[-1])


def test_rsi_all_gains_is_100():
    closes = pd.Series([100 + i for i in range(20)])
    result = ta.rsi(closes, window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_zero():
    closes = pd.Series([100 - i for i in range(20)])
    result = ta.rsi(closes, window=14)
    assert result.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_flat_series_is_neutral():
    closes = pd.Series([100.0] * 20)
    result = ta.rsi(closes, window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_macd_returns_three_series_of_equal_length():
    closes = pd.Series(np.linspace(100, 150, 60))
    macd_line, signal_line, histogram = ta.macd(closes)
    assert len(macd_line) == len(signal_line) == len(histogram) == len(closes)
    assert histogram.iloc[-1] == pytest.approx(macd_line.iloc[-1] - signal_line.iloc[-1])


def test_atr_of_constant_true_range():
    high = pd.Series([102.0] * 20)
    low = pd.Series([98.0] * 20)
    close = pd.Series([100.0] * 20)
    result = ta.atr(high, low, close, window=14)
    assert result.iloc[-1] == pytest.approx(4.0)


def test_atr_multiple_from_sma_positive_when_price_above_average():
    close = pd.Series([100.0] * 60 + [130.0])
    high = close + 2
    low = close - 2
    multiple = ta.atr_multiple_from_sma(close, high, low, sma_window=50, atr_window=14)
    assert multiple is not None
    assert multiple > 0


def test_atr_multiple_from_sma_none_when_not_enough_data():
    close = pd.Series([100.0] * 10)
    assert ta.atr_multiple_from_sma(close, close, close, sma_window=50, atr_window=14) is None


def test_relative_volume_above_one_on_a_spike():
    volume = pd.Series([1000.0] * 20 + [3000.0])
    assert ta.relative_volume(volume, window=20) == pytest.approx(3.0)


def test_relative_volume_none_when_not_enough_history():
    assert ta.relative_volume(pd.Series([1000.0] * 5), window=20) is None


def test_pct_change_over():
    closes = pd.Series([100.0, 105.0, 110.0, 121.0])
    assert ta.pct_change_over(closes, 1) == pytest.approx(0.1)
    assert ta.pct_change_over(closes, 3) == pytest.approx(0.21)
    assert ta.pct_change_over(closes, 10) is None


def test_distance_to_rolling_extreme_high_is_zero_at_the_peak():
    closes = pd.Series([90.0, 95.0, 100.0])
    # min_periods=3 opts into a short window on purpose here, to test the
    # max/min math in isolation - the default (no min_periods) is exercised
    # below, where it must return None instead of quietly answering with
    # whatever's available (D11).
    assert ta.distance_to_rolling_extreme(closes, window=252, kind="high", min_periods=3) == pytest.approx(0.0)


def test_distance_to_rolling_extreme_low():
    closes = pd.Series([100.0, 90.0, 95.0])
    # latest close (95) vs the rolling min (90) -> +5.56%
    result = ta.distance_to_rolling_extreme(closes, window=252, kind="low", min_periods=3)
    assert result == pytest.approx(95 / 90 - 1)


def test_distance_to_rolling_extreme_none_when_window_is_not_full():
    # D11: a "52-week" (window=252) read used to silently fall back to
    # whatever shorter history was available and label it as if it were the
    # real annual figure - e.g. a 60-bar (~3 month) high reported as a 52-week
    # high. Without an explicit min_periods, it must be None instead.
    closes = pd.Series(np.linspace(90.0, 100.0, 60))
    assert ta.distance_to_rolling_extreme(closes, window=252, kind="high") is None
    assert ta.distance_to_rolling_extreme(closes, window=252, kind="low") is None


def test_distance_to_rolling_extreme_none_below_explicit_min_periods_too():
    closes = pd.Series(np.linspace(90.0, 100.0, 30))
    assert ta.distance_to_rolling_extreme(closes, window=252, kind="high", min_periods=60) is None


def test_classify_trend_uptrend_when_mas_are_stacked_bullishly():
    assert ta.classify_trend(price=110, sma20=105, sma50=100, sma200=90) == ta.TrendState.UPTREND


def test_classify_trend_downtrend_when_mas_are_stacked_bearishly():
    assert ta.classify_trend(price=80, sma20=85, sma50=90, sma200=100) == ta.TrendState.DOWNTREND


def test_classify_trend_sideways_when_mas_are_tangled():
    assert ta.classify_trend(price=100, sma20=95, sma50=105, sma200=98) == ta.TrendState.SIDEWAYS


def test_detect_recent_cross_golden():
    fast = pd.Series([10, 10, 10, 12, 14])
    slow = pd.Series([11, 11, 11, 11, 11])
    assert ta.detect_recent_cross(fast, slow, lookback=5) == "golden"


def test_detect_recent_cross_death():
    fast = pd.Series([12, 12, 12, 10, 8])
    slow = pd.Series([11, 11, 11, 11, 11])
    assert ta.detect_recent_cross(fast, slow, lookback=5) == "death"


def test_detect_recent_cross_none_when_no_crossover():
    fast = pd.Series([12, 12, 13, 13, 14])
    slow = pd.Series([11, 11, 11, 11, 11])
    assert ta.detect_recent_cross(fast, slow, lookback=5) is None


def test_detect_recent_cross_detects_a_cross_exactly_lookback_bars_ago():
    # D5: a prior off-by-one pulled `lookback` *points* (lookback-1
    # transitions) instead of `lookback` *transitions* - so lookback=5 only
    # ever really checked the last 4 bar-to-bar changes, and a cross that
    # happened exactly 5 bars back was silently missed. diff = fast - slow is
    # negative for 5 bars, then positive for 5 bars: the golden cross happens
    # 4 bars before the last one, i.e. is only visible if the window actually
    # spans lookback+1=6 points (5 transitions), not 5.
    diff = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    fast = pd.Series(diff)
    slow = pd.Series([0] * len(diff))
    assert ta.detect_recent_cross(fast, slow, lookback=5) == "golden"
    # Confirms the fix is about window width, not just "any lookback": with a
    # window too narrow to reach the transition at all, it's still None.
    assert ta.detect_recent_cross(fast, slow, lookback=3) is None


def test_detect_imminent_cross_death_when_gap_shrinks_toward_zero():
    # fast - slow shrinks linearly from 20 to 6 over 15 bars (slope -1): still
    # positive (fast above slow, a "golden" state) but heading to 0 in 6 bars.
    gap = list(range(20, 5, -1))  # 20, 19, ..., 6 (15 values)
    fast = pd.Series(gap, dtype=float)
    slow = pd.Series([0.0] * len(gap))
    result = ta.detect_imminent_cross(fast, slow)
    assert result is not None
    assert result.direction == "death"
    assert result.bars_until == 6
    assert result.r_squared == 1.0  # a perfectly straight line


def test_detect_imminent_cross_golden_when_gap_rises_toward_zero():
    gap = [-20.0 + i for i in range(15)]  # -20, -19, ..., -6
    fast = pd.Series(gap)
    slow = pd.Series([0.0] * len(gap))
    result = ta.detect_imminent_cross(fast, slow)
    assert result is not None
    assert result.direction == "golden"
    assert result.bars_until == 6


def test_detect_imminent_cross_none_with_insufficient_history():
    fast = pd.Series([12.0, 11.0, 10.0])
    slow = pd.Series([10.0, 10.0, 10.0])
    assert ta.detect_imminent_cross(fast, slow) is None


def test_detect_imminent_cross_none_when_diverging():
    # Gap growing *away* from zero - moving in the wrong direction to cross soon.
    gap = [5.0 + i for i in range(15)]  # 5, 6, ..., 19
    fast = pd.Series(gap)
    slow = pd.Series([0.0] * len(gap))
    assert ta.detect_imminent_cross(fast, slow) is None


def test_detect_imminent_cross_none_when_flat():
    # No trend at all in the gap - a constant series has zero variance to explain,
    # so R² can't clear the goodness-of-fit bar.
    fast = pd.Series([5.0] * 15)
    slow = pd.Series([0.0] * 15)
    assert ta.detect_imminent_cross(fast, slow) is None


def test_detect_imminent_cross_none_when_already_crossed():
    gap = list(range(14, -1, -1))  # 14, 13, ..., 0 - lands exactly on zero
    fast = pd.Series(gap, dtype=float)
    slow = pd.Series([0.0] * len(gap))
    assert ta.detect_imminent_cross(fast, slow) is None


def test_detect_imminent_cross_none_when_projection_too_far_out():
    gap = [100.0 - 0.5 * i for i in range(15)]  # converging, but far too slowly
    fast = pd.Series(gap)
    slow = pd.Series([0.0] * len(gap))
    assert ta.detect_imminent_cross(fast, slow) is None


def test_detect_engulfing_pattern_bullish():
    # Prior bar red (100 -> 95), current bar green and fully covers it (94 -> 102).
    open_ = pd.Series([100.0, 94.0])
    close = pd.Series([95.0, 102.0])
    assert ta.detect_engulfing_pattern(open_, close) == "bullish_engulfing"


def test_detect_engulfing_pattern_bearish():
    # Prior bar green (95 -> 100), current bar red and fully covers it (102 -> 94).
    open_ = pd.Series([95.0, 102.0])
    close = pd.Series([100.0, 94.0])
    assert ta.detect_engulfing_pattern(open_, close) == "bearish_engulfing"


def test_detect_engulfing_pattern_none_when_body_does_not_fully_cover_prior():
    # Current bar is green and bigger than the prior red bar's close, but its
    # open doesn't reach down to the prior close - not a full engulf.
    open_ = pd.Series([100.0, 97.0])
    close = pd.Series([95.0, 102.0])
    assert ta.detect_engulfing_pattern(open_, close) is None


def test_detect_engulfing_pattern_none_when_same_direction():
    # Two green bars in a row - nothing being reversed/engulfed.
    open_ = pd.Series([95.0, 100.0])
    close = pd.Series([100.0, 105.0])
    assert ta.detect_engulfing_pattern(open_, close) is None


def test_detect_engulfing_pattern_none_with_a_doji_prior_bar():
    # Prior bar has (essentially) no real body to engulf.
    open_ = pd.Series([100.0, 94.0])
    close = pd.Series([100.0, 102.0])
    assert ta.detect_engulfing_pattern(open_, close) is None


def test_detect_engulfing_pattern_none_with_insufficient_history():
    open_ = pd.Series([100.0])
    close = pd.Series([95.0])
    assert ta.detect_engulfing_pattern(open_, close) is None


def test_support_resistance_levels_finds_pivots_around_current_price():
    # A clean V-shape then a bounce: pivot low at the trough, pivot high before it.
    prices = [100, 105, 110, 108, 104, 98, 94, 90, 94, 98, 104, 108, 106]
    close = pd.Series(prices, dtype=float)
    high = close + 1
    low = close - 1

    levels = ta.support_resistance_levels(high, low, close, left=2, right=2)

    assert any(lv.kind == "resistance" for lv in levels)
    assert any(lv.kind == "support" for lv in levels)
    for level in levels:
        if level.kind == "resistance":
            assert level.price > close.iloc[-1]
        else:
            assert level.price < close.iloc[-1]


def test_support_resistance_levels_empty_series():
    empty = pd.Series(dtype=float)
    assert ta.support_resistance_levels(empty, empty, empty) == []


def test_rolling_extreme_price_high_and_low():
    closes = pd.Series([90.0, 110.0, 95.0])
    assert ta.rolling_extreme_price(closes, window=252, kind="high", min_periods=3) == pytest.approx(110.0)
    assert ta.rolling_extreme_price(closes, window=252, kind="low", min_periods=3) == pytest.approx(90.0)


def test_rolling_extreme_price_none_when_window_is_not_full():
    # D11: same guard as distance_to_rolling_extreme - this backs the +2 point
    # "Movimiento confirmado" factor in recommendation_engine.py, the single
    # most validated factor in the checklist per docs/quant_methodology.md, so
    # silently truncating its input window was the highest-stakes instance of
    # this bug.
    closes = pd.Series(np.linspace(90.0, 110.0, 60))
    assert ta.rolling_extreme_price(closes, window=252, kind="high") is None
    assert ta.rolling_extreme_price(closes, window=252, kind="low") is None


def test_dmi_dominant_plus_di_in_a_clean_uptrend():
    n = 60
    close = pd.Series(100 + np.arange(n) * 0.8)
    high = close + 1
    low = close - 1
    plus_di, minus_di = ta.dmi(high, low, close)
    assert plus_di.iloc[-1] > minus_di.iloc[-1]


def test_adx_higher_for_a_clean_trend_than_a_choppy_range():
    n = 80
    trending_close = pd.Series(100 + np.arange(n) * 0.8)
    choppy_close = pd.Series(100 + 5 * np.sin(np.arange(n) / 2))
    trending_adx = ta.adx(trending_close + 1, trending_close - 1, trending_close).iloc[-1]
    choppy_adx = ta.adx(choppy_close + 1, choppy_close - 1, choppy_close).iloc[-1]
    assert trending_adx > choppy_adx


def test_mansfield_rs_positive_when_outperforming_benchmark():
    n = 260
    stock = pd.Series(100 * (1.006 ** np.arange(n)))
    benchmark = pd.Series(100 * (1.001 ** np.arange(n)))
    result = ta.mansfield_rs(stock, benchmark, window=200)
    assert result.iloc[-1] > 0


def test_mansfield_rs_negative_when_underperforming_benchmark():
    n = 260
    stock = pd.Series(100 * (1.001 ** np.arange(n)))
    benchmark = pd.Series(100 * (1.006 ** np.arange(n)))
    result = ta.mansfield_rs(stock, benchmark, window=200)
    assert result.iloc[-1] < 0


def test_rs_raw_score_positive_for_a_steady_riser():
    closes = pd.Series(100 * (1.002 ** np.arange(260)))
    score = ta.rs_raw_score(closes)
    assert score is not None
    assert score > 0


def test_rs_raw_score_none_when_insufficient_history():
    assert ta.rs_raw_score(pd.Series([100.0] * 50)) is None


def test_sma_slope_positive_true_for_rising_series():
    series = pd.Series(np.arange(40) * 1.0)
    assert ta.sma_slope_positive(series, lookback=25) is True


def test_sma_slope_positive_false_for_falling_series():
    series = pd.Series(100 - np.arange(40) * 1.0)
    assert ta.sma_slope_positive(series, lookback=25) is False


def test_sma_slope_positive_none_when_not_enough_data():
    assert ta.sma_slope_positive(pd.Series([1.0, 2.0]), lookback=25) is None


def test_classify_stage_stage2_when_ma_rising_and_price_above():
    sma_trend = pd.Series(100 + np.arange(150) * 0.3)
    assert ta.classify_stage(price=float(sma_trend.iloc[-1]) + 5, sma_trend=sma_trend) == ta.Stage.STAGE_2


def test_classify_stage_stage4_when_ma_falling_and_price_below():
    sma_trend = pd.Series(200 - np.arange(150) * 0.3)
    assert ta.classify_stage(price=float(sma_trend.iloc[-1]) - 5, sma_trend=sma_trend) == ta.Stage.STAGE_4


def test_classify_stage_stage1_basing_after_a_prior_decline():
    decline = 200 - np.arange(100) * 0.5  # 200 -> 150.5
    flat = np.full(30, decline[-1]) + np.linspace(0, 0.2, 30)
    sma_trend = pd.Series(np.concatenate([decline, flat]))
    assert ta.classify_stage(price=float(sma_trend.iloc[-1]), sma_trend=sma_trend) == ta.Stage.STAGE_1


def test_classify_stage_stage3_topping_after_a_prior_advance():
    advance = 100 + np.arange(100) * 0.5  # 100 -> 149.5
    flat = np.full(30, advance[-1]) + np.linspace(0, 0.2, 30)
    sma_trend = pd.Series(np.concatenate([advance, flat]))
    assert ta.classify_stage(price=float(sma_trend.iloc[-1]), sma_trend=sma_trend) == ta.Stage.STAGE_3


def test_classify_stage_none_when_not_enough_data():
    assert ta.classify_stage(price=100, sma_trend=pd.Series([100.0, 101.0])) is None


def test_minervini_checklist_all_pass():
    criteria = ta.minervini_checklist(
        price=150,
        sma50=140,
        sma150=130,
        sma200=120,
        sma200_trending_up=True,
        price_52w_low=100,
        price_52w_high=160,
        rs_rating=85,
    )
    assert all(criteria.values())


def test_minervini_checklist_fails_rs_rating_criterion():
    criteria = ta.minervini_checklist(
        price=150,
        sma50=140,
        sma150=130,
        sma200=120,
        sma200_trending_up=True,
        price_52w_low=100,
        price_52w_high=160,
        rs_rating=40,
    )
    assert criteria["rs_rating_70_plus"] is False
    assert not all(criteria.values())


def test_obv_rises_on_up_days_and_falls_on_down_days():
    close = pd.Series([100.0, 101.0, 99.0, 102.0])
    volume = pd.Series([1000.0, 500.0, 300.0, 700.0])
    result = ta.obv(close, volume)
    # day0: no prior diff -> 0 contribution; day1 up +500; day2 down -300; day3 up +700
    assert result.tolist() == pytest.approx([0.0, 500.0, 200.0, 900.0])


def test_rolling_position_in_range_at_extremes():
    series = pd.Series([10.0, 20.0, 15.0, 20.0, 10.0])
    result = ta.rolling_position_in_range(series, window=5)
    # last value (10.0) is the min of the whole window -> position 0
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rolling_position_in_range_flat_series_is_nan_span_safe():
    series = pd.Series([50.0] * 10)
    result = ta.rolling_position_in_range(series, window=5)
    # zero span (no range at all) must not raise or produce inf - clipped/NaN-safe
    assert not np.isinf(result.dropna()).any()


def test_obv_divergence_bearish_when_price_leads_obv_at_the_top():
    # Price grinds to a new range high on mostly-thin volume, with occasional
    # heavier-volume down days - OBV lags well behind the price high, a
    # textbook Wyckoff distribution warning.
    n = 40
    prices = [100.0]
    vols = [1000.0]
    for i in range(1, n):
        if i % 4 == 0:
            prices.append(prices[-1] - 0.3)
            vols.append(3000.0)  # big volume on the down day
        else:
            prices.append(prices[-1] + 0.4)
            vols.append(300.0)  # thin volume on up days
    close = pd.Series(prices)
    volume = pd.Series(vols)
    result = ta.obv_divergence(close, volume, window=20)
    assert result == "bearish"


def test_obv_divergence_none_when_price_and_volume_move_together():
    n = 40
    prices = [100.0]
    vols = [500.0]
    for _ in range(1, n):
        prices.append(prices[-1] + 0.5)
        vols.append(500.0)  # steady volume, no divergence signal either way
    close = pd.Series(prices)
    volume = pd.Series(vols)
    result = ta.obv_divergence(close, volume, window=20)
    assert result is None


def test_obv_divergence_none_when_too_little_history():
    close = pd.Series([100.0, 101.0, 102.0])
    volume = pd.Series([500.0, 600.0, 700.0])
    assert ta.obv_divergence(close, volume, window=20) is None


@pytest.mark.parametrize(
    "level,expected",
    [
        (None, "desconocido"),
        (8, "complacencia"),
        (15, "normal"),
        (25, "miedo elevado"),
        (35, "pánico"),
        (50, "crisis"),
    ],
)
def test_vix_regime_bands(level, expected):
    # Moved here from market_context_service.py (2026-08) so both the live
    # "Contexto" dashboard and the recommendation engine's market-regime gate
    # can share one definition without a circular import.
    assert ta.vix_regime(level) == expected


def test_market_regime_inputs_detects_benchmark_below_its_200sma():
    n = 300
    # A benchmark that spent its first 250 bars climbing (building a real
    # SMA200), then rolled over hard for the last 50 - clearly below its own
    # 200-day average by the end.
    rise = 100 + np.arange(250) * 0.3
    decline = rise[-1] - np.arange(1, 51) * 1.0
    benchmark_close = pd.Series(np.concatenate([rise, decline]))
    vix_close = pd.Series([32.0] * n)  # "pánico" band

    market_trend, vix_label = ta.market_regime_inputs(benchmark_close, vix_close)

    assert market_trend == ta.TrendState.DOWNTREND
    assert vix_label == "pánico"


def test_market_regime_inputs_none_when_no_data_supplied():
    market_trend, vix_label = ta.market_regime_inputs(None, None)
    assert market_trend is None
    assert vix_label is None


def test_market_regime_inputs_none_when_insufficient_benchmark_history():
    short_benchmark = pd.Series(100 + np.arange(50) * 0.5)
    market_trend, vix_label = ta.market_regime_inputs(short_benchmark, None)
    assert market_trend is None
    assert vix_label is None
