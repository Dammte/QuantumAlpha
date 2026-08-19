import numpy as np
import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta


def _read(
    timeframe: str = "daily",
    trend: ta.TrendState = ta.TrendState.SIDEWAYS,
    stage: ta.Stage | None = None,
    ma_cross_50_200: str | None = None,
    macd_cross: str | None = None,
) -> mtf.TimeframeRead:
    """Builds a TimeframeRead with every field neutral/None except the ones a
    test cares about - combine_timeframes only looks at trend/stage (via
    _is_bullish/_is_bearish) and the two cross fields, so those are the only
    ones worth parameterizing here."""
    return mtf.TimeframeRead(
        timeframe=timeframe,
        trend=trend,
        stage=stage,
        ma_cross_50_200=ma_cross_50_200,
        ma_cross_20_50=None,
        cross_quality_20_50=None,
        imminent_cross_50_200=None,
        imminent_cross_20_50=None,
        macd_cross=macd_cross,
        macd_histogram_turning=None,
        rsi14=None,
        adx14=None,
        di_bias=None,
        price_vs_sma20=None,
        price_vs_sma50=None,
        price_vs_sma200=None,
        bars_since_cross=None,
    )


# --- combine_timeframes: the alignment hierarchy, tested against hand-built
# TimeframeReads so every branch is exercised with an obviously-correct
# expected answer, independent of whether a real price series happens to hit
# the exact SMA/stage boundary needed to reach that branch.


def test_bearish_aligned_when_both_timeframes_trend_down():
    weekly = _read("weekly", trend=ta.TrendState.DOWNTREND)
    daily = _read("daily", trend=ta.TrendState.DOWNTREND)
    alignment, score, conflicts = mtf.combine_timeframes(weekly, daily)
    assert alignment == "bearish_aligned"
    assert score == -1.0
    assert conflicts == []


def test_bearish_aligned_via_weinstein_stage_even_without_ma_trend():
    # Stage 4 (declive) counts as bearish on its own, independent of the
    # MA-ordering trend read - two different lenses on the same fact.
    weekly = _read("weekly", stage=ta.Stage.STAGE_4)
    daily = _read("daily", stage=ta.Stage.STAGE_4)
    alignment, score, _ = mtf.combine_timeframes(weekly, daily)
    assert alignment == "bearish_aligned"
    assert score == -1.0


def test_bullish_aligned_when_both_timeframes_trend_up():
    weekly = _read("weekly", trend=ta.TrendState.UPTREND)
    daily = _read("daily", trend=ta.TrendState.UPTREND)
    alignment, score, conflicts = mtf.combine_timeframes(weekly, daily)
    assert alignment == "bullish_aligned"
    assert score == 1.0
    assert conflicts == []


def test_conflicted_and_scored_negative_when_weekly_bearish_daily_bullish():
    # The case the whole exercise is about: a daily bounce inside a bigger
    # weekly downtrend must score *against* the setup, not neutrally.
    weekly = _read("weekly", trend=ta.TrendState.DOWNTREND)
    daily = _read("daily", trend=ta.TrendState.UPTREND)
    alignment, score, conflicts = mtf.combine_timeframes(weekly, daily)
    assert alignment == "conflicted"
    assert score < 0
    assert any("rebote" in c for c in conflicts)


def test_conflicted_and_scored_positive_when_weekly_bullish_daily_bearish():
    weekly = _read("weekly", trend=ta.TrendState.UPTREND)
    daily = _read("daily", trend=ta.TrendState.DOWNTREND)
    alignment, score, conflicts = mtf.combine_timeframes(weekly, daily)
    assert alignment == "conflicted"
    assert score > 0
    assert any("corrección" in c for c in conflicts)


def test_transitioning_when_neither_timeframe_shows_a_clear_trend():
    weekly = _read("weekly", trend=ta.TrendState.SIDEWAYS)
    daily = _read("daily", trend=ta.TrendState.SIDEWAYS)
    alignment, score, conflicts = mtf.combine_timeframes(weekly, daily)
    assert alignment == "transitioning"
    assert score == 0.0
    assert conflicts == []


def test_conflict_flagged_when_weekly_death_cross_hasnt_reached_daily_yet():
    weekly = _read("weekly", trend=ta.TrendState.DOWNTREND, ma_cross_50_200="death")
    daily = _read("daily", trend=ta.TrendState.DOWNTREND, ma_cross_50_200=None)
    _, _, conflicts = mtf.combine_timeframes(weekly, daily)
    assert any("Cruce de medias bajista" in c and "semanal" in c for c in conflicts)


def test_conflict_flagged_when_weekly_macd_bearish_and_daily_macd_bullish():
    weekly = _read("weekly", trend=ta.TrendState.DOWNTREND, macd_cross="bearish")
    daily = _read("daily", trend=ta.TrendState.DOWNTREND, macd_cross="bullish")
    _, _, conflicts = mtf.combine_timeframes(weekly, daily)
    assert any("MACD semanal" in c for c in conflicts)


def test_daily_only_fallback_is_bullish_when_weekly_is_unavailable():
    daily = _read("daily", trend=ta.TrendState.UPTREND)
    alignment, score, conflicts = mtf.combine_timeframes(None, daily)
    assert alignment == "bullish_aligned"
    assert score == 1.0
    assert conflicts == []


def test_daily_only_fallback_is_bearish_when_weekly_is_unavailable():
    daily = _read("daily", trend=ta.TrendState.DOWNTREND)
    alignment, score, _ = mtf.combine_timeframes(None, daily)
    assert alignment == "bearish_aligned"
    assert score == -1.0


def test_daily_only_fallback_is_transitioning_when_weekly_is_unavailable():
    daily = _read("daily", trend=ta.TrendState.SIDEWAYS)
    alignment, score, _ = mtf.combine_timeframes(None, daily)
    assert alignment == "transitioning"
    assert score == 0.0


# --- analyze_multi_timeframe: end-to-end plumbing (resample -> closed_bars ->
# indicators -> combine_timeframes) against long, clean synthetic price
# series, where the direction is unambiguous on both timeframes at once.


def _ohlcv_df(n_days: int, closes) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    close = pd.Series(closes, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([1_000_000.0] * n_days, index=dates),
        }
    )


def test_analyze_multi_timeframe_end_to_end_bullish_aligned():
    # ~4.3 years of business days - enough for a real weekly SMA200 (~220
    # weekly bars), not just a daily read.
    n = 1100
    df = _ohlcv_df(n, 100 + np.arange(n) * 0.3)
    result = mtf.analyze_multi_timeframe(df)

    assert result.weekly is not None
    assert result.daily.trend == ta.TrendState.UPTREND
    assert result.weekly.trend == ta.TrendState.UPTREND
    assert result.alignment == "bullish_aligned"
    assert result.alignment_score == 1.0
    assert result.intraday is None  # not populated yet - see D1/Fase 1.2


def test_analyze_multi_timeframe_end_to_end_bearish_aligned():
    n = 1100
    df = _ohlcv_df(n, 500 - np.arange(n) * 0.3)
    result = mtf.analyze_multi_timeframe(df)

    assert result.weekly is not None
    assert result.daily.trend == ta.TrendState.DOWNTREND
    assert result.weekly.trend == ta.TrendState.DOWNTREND
    assert result.alignment == "bearish_aligned"
    assert result.alignment_score == -1.0


def test_analyze_multi_timeframe_weekly_is_none_with_too_little_history():
    # A handful of daily bars can't even form a couple of closed weekly bars.
    df = _ohlcv_df(5, [100.0, 101.0, 102.0, 101.5, 103.0])
    result = mtf.analyze_multi_timeframe(df)
    assert result.weekly is None
    assert result.daily is not None


def test_analyze_multi_timeframe_weekly_resample_has_no_network_cost():
    # Not a real assertion of "no network call" (this module makes none by
    # construction - no imports of any HTTP/data-provider client), but a
    # structural guard: analyze_multi_timeframe must only ever be handed a
    # DataFrame, never a ticker/service, so nobody can accidentally wire in a
    # per-call fetch here later.
    import inspect

    params = inspect.signature(mtf.analyze_multi_timeframe).parameters
    assert list(params) == ["daily_df"]
    assert params["daily_df"].annotation in ("pd.DataFrame", pd.DataFrame)
