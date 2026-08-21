import numpy as np
import pandas as pd

from app.services import ticker_analysis_service as tas

_SeriesQuintet = tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]


def _series(close: np.ndarray, wiggle: float = 1.0) -> _SeriesQuintet:
    # A real DatetimeIndex, ending well before "today" - compute_core_signals
    # now derives weekly bars (multi_timeframe.analyze_multi_timeframe) off
    # this index via technical_analysis.closed_bars/resample_ohlcv, which
    # both need real dates, not a bare positional RangeIndex. Fixed in the
    # past (not ending "today") keeps these tests' bars all settled, same as
    # the plain int index they used before (is_intraday_snapshot was always
    # False for that shape of index too).
    index = pd.bdate_range(end=pd.Timestamp("2024-01-01"), periods=len(close))
    close_s = pd.Series(close, index=index)
    return close_s, close_s + wiggle, close_s - wiggle, pd.Series([1_000_000.0] * len(close), index=index), close_s


def test_none_when_not_enough_bars():
    close, high, low, volume, open_ = _series(np.array([100.0] * 10))
    assert tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None) is None


def test_returns_a_fully_populated_result_for_an_uptrend():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=85)

    assert signals is not None
    assert signals.trend == tas.ta.TrendState.UPTREND
    assert signals.rs_rating == 85
    assert signals.recommendation is not None
    assert signals.garch is not None
    # Note: a perfectly deterministic straight-line series (no noise at all,
    # unlike a real market) can leave Markov/backtest inputs degenerate, so
    # markov/backtest legitimately being None here isn't asserted either way -
    # see test_ticker_analysis_api.py for full-suite coverage against the
    # (noisy, realistic) fake random-walk price data.


def test_rs_rating_passed_through_unchanged():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=42)
    assert signals.rs_rating == 42


def test_mansfield_rs_is_none_without_a_benchmark():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals.mansfield_rs is None


def test_mansfield_rs_is_computed_when_a_benchmark_is_given():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    benchmark = pd.Series(100 + np.arange(260) * 0.1, index=close.index)  # mansfield_rs inner-joins by index
    signals = tas.compute_core_signals(close, high, low, volume, open_, benchmark, rs_rating=None)
    assert signals.mansfield_rs is not None


def test_position_sizing_only_present_for_a_comprar_verdict_with_a_trade_setup():
    close, high, low, volume, open_ = _series(200 - np.arange(260) * 0.3)  # clear downtrend -> "evitar"
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals.recommendation.verdict != "comprar"
    assert signals.position_sizing is None


def test_multi_timeframe_is_always_populated():
    # Segunda auditoría, Bloque 2: before this, ticker_analysis_service.py
    # never referenced analyze_multi_timeframe/closed_bars at all - every
    # signal was read off the live/possibly-still-forming last bar with no
    # weekly/confirmed counterpart exposed anywhere.
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.multi_timeframe is not None
    assert signals.multi_timeframe.daily is not None


def test_confirmed_recommendation_is_none_when_the_last_bar_is_already_settled():
    # These fixtures end well in the past (see _series) - is_intraday_snapshot
    # is False, so there is nothing to separate the live verdict from.
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.is_intraday_snapshot is False
    assert signals.confirmed_recommendation is None


def test_confirmed_recommendation_is_populated_when_the_last_bar_is_still_forming():
    # A series whose last bar is dated *today* - the live read is real but not
    # repaint-proof, so confirmed_recommendation must be filled in from
    # technical_analysis.closed_bars instead of silently staying None.
    n = 260
    # Calendar days (not bdate_range) so the last bar lands on "today"
    # regardless of which weekday the test happens to run on.
    index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    close_s = pd.Series(100 + np.arange(n) * 0.4, index=index)
    high, low = close_s + 1.0, close_s - 1.0
    volume = pd.Series([1_000_000.0] * n, index=index)
    signals = tas.compute_core_signals(close_s, high, low, volume, close_s, None, rs_rating=None)
    assert signals is not None
    assert signals.is_intraday_snapshot is True
    assert signals.confirmed_recommendation is not None


def test_triple_barrier_backtest_is_populated_with_enough_history():
    # Segunda auditoría, Bloque 2: run_triple_barrier_backtest must actually
    # have a caller - before this, it existed, was tested in isolation, and
    # was never wired into "Analizar activo" at all. Needs more history than
    # the other fixtures here (WARMUP_BARS=260 plus the horizon) to produce a
    # real (non-None) result, at the fixed 21-day horizon (never the Monte
    # Carlo preset - see TRIPLE_BARRIER_HORIZON_DAYS).
    close, high, low, volume, open_ = _series(100 + np.arange(400) * 0.4)
    signals = tas.compute_core_signals(
        close, high, low, volume, open_, None, rs_rating=None, include_triple_barrier_backtest=True
    )
    assert signals is not None
    assert signals.triple_barrier_backtest is not None
    assert signals.triple_barrier_backtest.horizon_days == tas.TRIPLE_BARRIER_HORIZON_DAYS


def test_triple_barrier_backtest_is_skipped_by_default():
    # Measured at ~3x this function's own cost - portfolio_risk_service.py
    # and premium_watchlist_service.py (which run this per held position /
    # per candidate on every cache refresh) must not pay for a field neither
    # of those views shows. Only TickerAnalysisService.analyze() opts in.
    close, high, low, volume, open_ = _series(100 + np.arange(400) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.triple_barrier_backtest is None


def test_52_week_range_factor_never_fires_with_only_60_bars():
    # D11: rolling_extreme_price/distance_to_rolling_extreme now require the
    # full 252-bar window by default - a ticker with only 60 bars (the
    # minimum MIN_BARS_REQUIRED accepts at all) must not have its "52-week
    # range" answered with whatever 60 days happen to be available. This
    # backs the single most validated factor in the checklist (see
    # docs/quant_methodology.md), so it's the highest-stakes instance of D11.
    close, high, low, volume, open_ = _series(100 + np.arange(60) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.dist_52w_high is None
    assert signals.dist_52w_low is None
    range_factor = "Movimiento confirmado: precio 25%+ sobre su mínimo anual y dentro del 25% de su máximo anual"
    assert not any(f.triggered for f in signals.recommendation.factors if f.label == range_factor)
