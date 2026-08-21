import numpy as np
import pandas as pd
import pytest

from app.services import market_screener_service as mss


def _df(closes, highs=None, lows=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    index = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(closes, index=index)
    high = pd.Series(highs, index=index) if highs is not None else close + 1.0
    low = pd.Series(lows, index=index) if lows is not None else close - 1.0
    volume = pd.Series(volumes, index=index) if volumes is not None else pd.Series([1_000_000.0] * n, index=index)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


# --- Segunda auditoría, Bloque 3: new cross-sectional-score inputs -------------


def test_build_raw_atr_ratio_below_one_when_the_range_has_contracted():
    # A wide daily range for 60 bars, then a much narrower one for the last
    # 20 - volatility has genuinely contracted relative to its own recent
    # (50-bar) average.
    n = 90
    closes = [100.0] * n
    highs = [105.0] * 70 + [100.4] * 20
    lows = [95.0] * 70 + [99.6] * 20
    df = _df(closes, highs=highs, lows=lows)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.atr_ratio_50d is not None
    assert raw.atr_ratio_50d < 1.0


def test_build_raw_atr_ratio_above_one_when_the_range_has_expanded():
    n = 90
    closes = [100.0] * n
    highs = [100.4] * 70 + [110.0] * 20
    lows = [99.6] * 70 + [90.0] * 20
    df = _df(closes, highs=highs, lows=lows)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.atr_ratio_50d is not None
    assert raw.atr_ratio_50d > 1.0


def test_build_raw_range_position_20d_near_one_at_a_20_day_high():
    n = 80
    closes = list(100 + np.arange(n) * 0.5)  # steady climb - today is the 20-day high
    df = _df(closes)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.range_position_20d == pytest.approx(1.0)


def test_build_raw_range_position_20d_near_zero_at_a_20_day_low():
    n = 80
    closes = list(200 - np.arange(n) * 0.5)  # steady decline - today is the 20-day low
    df = _df(closes)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.range_position_20d == pytest.approx(0.0)


def test_build_raw_mansfield_rs_4w_uses_a_20_session_window():
    n = 260
    close = pd.Series(100 + np.arange(n) * 0.3, index=pd.bdate_range("2020-01-01", periods=n))
    benchmark = pd.Series(100 + np.arange(n) * 0.05, index=close.index)  # outperforming the benchmark
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": [1_000_000.0] * n},
        index=close.index,
    )
    raw = mss._build_raw("TEST", "Tecnología", None, df, benchmark)
    assert raw is not None
    assert raw.mansfield_rs_4w is not None
    assert raw.mansfield_rs is not None
    # Different windows (20 vs 200 sessions) on a steadily-outperforming
    # series must not coincidentally produce the exact same reading.
    assert raw.mansfield_rs_4w != raw.mansfield_rs


def test_build_raw_relative_volume_trend_positive_when_volume_has_been_building():
    n = 60
    closes = [100.0] * n
    volumes = [1_000_000.0] * (n - 5) + [3_000_000.0] * 5  # a recent, sustained pickup
    df = _df(closes, volumes=volumes)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.relative_volume_trend is not None
    assert raw.relative_volume_trend > 0


def test_build_raw_relative_volume_trend_negative_when_volume_has_been_fading():
    n = 60
    closes = [100.0] * n
    volumes = [3_000_000.0] * (n - 5) + [1_000_000.0] * 5
    df = _df(closes, volumes=volumes)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.relative_volume_trend is not None
    assert raw.relative_volume_trend < 0


def test_build_raw_atr_multiple_sma21_differs_from_the_default_50_day_one():
    # A sharp recent run-up: the 21-day SMA is much closer to today's price
    # than the 50-day one, so the ATR-multiple-from-SMA reads differently
    # for each - if this function were still hardcoded to sma_window=50, the
    # 21-day field would just silently duplicate the existing one.
    closes = [100.0] * 240 + list(100 + np.arange(20) * 2.0)
    df = _df(closes)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.atr_multiple_sma21 is not None
    assert raw.atr_multiple is not None
    assert raw.atr_multiple_sma21 != raw.atr_multiple
