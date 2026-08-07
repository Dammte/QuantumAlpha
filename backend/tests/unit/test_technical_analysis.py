import numpy as np
import pandas as pd
import pytest

from app.services import technical_analysis as ta


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
    assert ta.distance_to_rolling_extreme(closes, window=252, kind="high") == pytest.approx(0.0)


def test_distance_to_rolling_extreme_low():
    closes = pd.Series([100.0, 90.0, 95.0])
    # latest close (95) vs the rolling min (90) -> +5.56%
    assert ta.distance_to_rolling_extreme(closes, window=252, kind="low") == pytest.approx(95 / 90 - 1)


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
    assert ta.rolling_extreme_price(closes, window=252, kind="high") == pytest.approx(110.0)
    assert ta.rolling_extreme_price(closes, window=252, kind="low") == pytest.approx(90.0)


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
