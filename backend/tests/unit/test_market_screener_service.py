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


# --- "Tendencia" screener, corto/mediano plazo (ago 2026): ma_cross_short /
# imminent_cross_short_term (SMA{FAST_MA_PERIOD}/SMA50, weeks not months) ----


def _reversal_closes(rally_bars: int) -> list[float]:
    """A decline (60 bars) followed by a rally, truncated to `rally_bars` of
    that rally - the fast MA (SMA21) is below the slow one (SMA50) after the
    decline, then catches up and eventually crosses above it during the
    rally. Varying `rally_bars` picks a point either before the actual cross
    (still converging - imminent_cross_short_term's territory) or after it
    (already happened - ma_cross_short's territory)."""
    decline = 100 - np.arange(60) * 0.5
    rally = decline[-1] + np.arange(1, rally_bars + 1) * 0.8
    return list(np.concatenate([decline, rally]))


def test_build_raw_ma_cross_short_detects_a_confirmed_golden_cross():
    # 81 bars total: the SMA21/SMA50 golden cross actually completes at bar
    # 79 (verified against ta.detect_recent_cross directly) - well within
    # ma_cross_short's own lookback=5 of this series' last bar (80).
    df = _df(_reversal_closes(rally_bars=21))
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.ma_cross_short == "golden"
    # Already happened - not also flagged as still "upcoming".
    assert raw.imminent_cross_short_term is None


def test_build_raw_imminent_cross_short_term_projects_a_golden_cross_before_it_happens():
    # Same reversal, cut 5 bars earlier (76 bars total) - well before the
    # actual cross at bar 79, but the SMA21/SMA50 gap is already converging
    # in a straight enough line to project it.
    df = _df(_reversal_closes(rally_bars=16))
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.ma_cross_short is None  # not confirmed yet
    assert raw.imminent_cross_short_term is not None
    assert raw.imminent_cross_short_term.direction == "golden"


def test_build_raw_ma_cross_short_and_imminent_cross_none_with_no_convergence():
    df = _df([100.0] * 60)
    raw = mss._build_raw("TEST", "Tecnología", None, df, None)
    assert raw is not None
    assert raw.ma_cross_short is None
    assert raw.imminent_cross_short_term is None


# --- get_trend_breadth / get_trend_detail: corto plazo + "más precisos" -------


def _snapshot(
    ticker: str,
    *,
    trend=mss.ta.TrendState.UPTREND,
    rs_rating=50,
    atr_multiple=1.0,
    ma_cross_short=None,
    imminent_cross_short_term=None,
    stage=None,
    minervini_pass=False,
    adx14=None,
):
    from app.domain.models.ticker_snapshot import TickerSnapshot

    return TickerSnapshot(
        ticker=ticker,
        sector="Tecnología",
        industry=None,
        cap_tier="large",
        price=100.0,
        change_1d=0.0,
        change_1w=0.0,
        change_1m=0.0,
        change_3m=0.0,
        change_6m=0.0,
        change_1y=0.0,
        volume=1_000_000.0,
        relative_volume=1.0,
        rsi14=50.0,
        sma20=95.0,
        sma50=90.0,
        sma150=85.0,
        sma200=80.0,
        dist_52w_high=-0.1,
        dist_52w_low=0.2,
        atr_multiple=atr_multiple,
        adx14=adx14,
        plus_di=20.0,
        minus_di=15.0,
        mansfield_rs=0.0,
        trend=trend,
        stage=stage,
        ma_cross=None,
        minervini_score=8 if minervini_pass else 0,
        minervini_pass=minervini_pass,
        rs_rating=rs_rating,
        ma_cross_short=ma_cross_short,
        imminent_cross_short_term=imminent_cross_short_term,
    )


def test_get_trend_breadth_golden_and_death_counts_use_the_short_term_pair():
    snapshots = [
        _snapshot("A", ma_cross_short="golden"),
        _snapshot("B", ma_cross_short="death"),
        _snapshot("C", ma_cross_short=None),
    ]
    breadth = mss.get_trend_breadth(snapshots)
    assert breadth.golden_crosses == 1
    assert breadth.death_crosses == 1


def test_get_trend_breadth_counts_imminent_crosses_by_direction():
    imminent_golden = mss.ta.ImminentCross(direction="golden", bars_until=5, r_squared=0.8)
    imminent_death = mss.ta.ImminentCross(direction="death", bars_until=3, r_squared=0.9)
    snapshots = [
        _snapshot("A", imminent_cross_short_term=imminent_golden),
        _snapshot("B", imminent_cross_short_term=imminent_death),
        _snapshot("C", imminent_cross_short_term=None),
    ]
    breadth = mss.get_trend_breadth(snapshots)
    assert breadth.count_imminent_golden == 1
    assert breadth.count_imminent_death == 1


def test_get_trend_detail_imminent_cross_group_sorted_by_soonest_first():
    soon = mss.ta.ImminentCross(direction="golden", bars_until=2, r_squared=0.8)
    later = mss.ta.ImminentCross(direction="death", bars_until=8, r_squared=0.7)
    snapshots = [
        _snapshot("LATER", imminent_cross_short_term=later),
        _snapshot("SOON", imminent_cross_short_term=soon),
        _snapshot("NONE", imminent_cross_short_term=None),
    ]
    detail = mss.get_trend_detail(snapshots)
    assert [s.ticker for s in detail["imminent_cross"]] == ["SOON", "LATER"]


def test_get_trend_detail_uptrend_deprioritizes_an_overextended_name_without_hiding_it():
    # HIGHRS has the best RS Rating but is already parabolic (atr_multiple > 4)
    # - "más precisos" (ago 2026): it should still appear, just not crowd out
    # a healthier trend candidate at the top of the list.
    snapshots = [
        _snapshot("HIGHRS_EXTENDED", rs_rating=95, atr_multiple=5.0),
        _snapshot("HEALTHY", rs_rating=80, atr_multiple=1.5),
    ]
    detail = mss.get_trend_detail(snapshots)
    assert [s.ticker for s in detail["uptrend"]] == ["HEALTHY", "HIGHRS_EXTENDED"]


def test_get_trend_detail_downtrend_ranking_is_unaffected_by_the_deprioritize_rule():
    # The overextension deprioritization only applies to the bullish/
    # "recommended" groups - downtrend keeps its plain rs_rating-ascending
    # order regardless of atr_multiple.
    snapshots = [
        _snapshot("WEAKEST", trend=mss.ta.TrendState.DOWNTREND, rs_rating=5, atr_multiple=5.0),
        _snapshot("LESS_WEAK", trend=mss.ta.TrendState.DOWNTREND, rs_rating=20, atr_multiple=1.0),
    ]
    detail = mss.get_trend_detail(snapshots)
    assert [s.ticker for s in detail["downtrend"]] == ["WEAKEST", "LESS_WEAK"]


def test_snapshot_from_dict_reconstructs_imminent_cross_as_a_real_object_not_a_dict():
    # A cache-reloaded snapshot's imminent_cross_short_term round-trips
    # through JSON as a plain dict via asdict() - _snapshot_from_dict must
    # turn it back into a real ImminentCross, or `.bars_until` access
    # downstream (get_trend_detail's own sort key) would crash on a dict.
    snapshot = _snapshot(
        "TEST", imminent_cross_short_term=mss.ta.ImminentCross(direction="golden", bars_until=4, r_squared=0.75)
    )
    data = mss._snapshot_to_dict(snapshot)
    assert isinstance(data["imminent_cross_short_term"], dict)  # confirms the round-trip actually exercises this

    rebuilt = mss._snapshot_from_dict(data)
    assert isinstance(rebuilt.imminent_cross_short_term, mss.ta.ImminentCross)
    assert rebuilt.imminent_cross_short_term.bars_until == 4


def test_snapshot_from_dict_handles_a_snapshot_cached_before_this_field_existed():
    snapshot = _snapshot("TEST")
    data = mss._snapshot_to_dict(snapshot)
    del data["imminent_cross_short_term"]  # simulates a pre-existing durable_cache row
    del data["ma_cross_short"]
    rebuilt = mss._snapshot_from_dict(data)
    assert rebuilt.imminent_cross_short_term is None
    assert rebuilt.ma_cross_short is None
