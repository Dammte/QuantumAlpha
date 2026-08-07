import numpy as np
import pandas as pd

from app.services import ticker_analysis_service as tas


def _series(close: np.ndarray, wiggle: float = 1.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close_s = pd.Series(close)
    return close_s, close_s + wiggle, close_s - wiggle, pd.Series([1_000_000.0] * len(close))


def test_none_when_not_enough_bars():
    close, high, low, volume = _series(np.array([100.0] * 10))
    assert tas.compute_core_signals(close, high, low, volume, None, rs_rating=None) is None


def test_returns_a_fully_populated_result_for_an_uptrend():
    close, high, low, volume = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, None, rs_rating=85)

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
    close, high, low, volume = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, None, rs_rating=42)
    assert signals.rs_rating == 42


def test_mansfield_rs_is_none_without_a_benchmark():
    close, high, low, volume = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, None, rs_rating=None)
    assert signals.mansfield_rs is None


def test_mansfield_rs_is_computed_when_a_benchmark_is_given():
    close, high, low, volume = _series(100 + np.arange(260) * 0.4)
    benchmark = pd.Series(100 + np.arange(260) * 0.1)
    signals = tas.compute_core_signals(close, high, low, volume, benchmark, rs_rating=None)
    assert signals.mansfield_rs is not None


def test_position_sizing_only_present_for_a_comprar_verdict_with_a_trade_setup():
    close, high, low, volume = _series(200 - np.arange(260) * 0.3)  # clear downtrend -> "evitar"
    signals = tas.compute_core_signals(close, high, low, volume, None, rs_rating=None)
    assert signals.recommendation.verdict != "comprar"
    assert signals.position_sizing is None
