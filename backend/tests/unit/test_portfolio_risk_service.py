import numpy as np
import pandas as pd

from app.services import portfolio_risk_service as prs


def _ohlc(close: np.ndarray, wiggle: float = 1.0) -> pd.DataFrame:
    close_s = pd.Series(close)
    return pd.DataFrame(
        {"close": close_s, "high": close_s + wiggle, "low": close_s - wiggle, "volume": 1_000_000.0}
    )


def test_none_when_not_enough_bars():
    df = _ohlc(np.array([100.0] * 10))
    assert prs.assess_position_risk("XYZ", df) is None


def test_exit_warning_for_a_clear_downtrend():
    close = 200 - np.arange(260) * 0.3
    df = _ohlc(close)
    result = prs.assess_position_risk("XYZ", df)
    assert result is not None
    assert result.signal == prs.EXIT_WARNING
    assert result.trend == "downtrend"
    assert any("bajista" in reason for reason in result.reasons)


def test_uptrend_never_flagged_as_exit_warning():
    close = 100 + np.arange(260) * 0.4
    df = _ohlc(close)
    result = prs.assess_position_risk("XYZ", df)
    assert result is not None
    assert result.trend == "uptrend"
    assert result.signal != prs.EXIT_WARNING


def test_add_candidate_when_uptrend_pulls_back_to_support():
    rise = 100 + np.arange(200) * 0.4  # steady climb to ~180
    dip = rise[-1] - np.array([0.0, 1.0, 1.8, 1.3, 0.6])  # brief, shallow pullback forms a swing low
    bounce = dip[-1] + np.arange(1, 4) * 0.4  # starts recovering, still close to the swing low
    close = np.concatenate([rise, dip, bounce])
    df = _ohlc(close)

    result = prs.assess_position_risk("XYZ", df)

    assert result is not None
    assert result.trend == "uptrend"
    assert result.nearest_support is not None
    assert abs(result.nearest_support.distance_pct) <= prs.PROXIMITY_THRESHOLD
    assert result.signal == prs.ADD_CANDIDATE


def test_strong_setup_near_resistance_is_add_candidate_not_watch():
    # The bug this module used to have: a leading stock making new highs sits
    # near a resistance/prior-high pivot by definition - the old logic flagged
    # ANY nearby level (support or resistance) as "watch" once it wasn't a
    # support-proximity add-candidate, so a stock that would score "comprar"
    # on the deep dive could show "vigilar" here for the same reason it's
    # strong. rs_rating=85 plus the uptrend/stage/support-adjacent factors is
    # enough to clear the recommendation engine's comprar threshold.
    rise = 100 + np.arange(220) * 0.4
    spike = rise[-1] + np.array([2.0, 4.0, 6.0, 4.5])
    pull_back = spike[-1] - np.array([0.5, 1.0])
    close = np.concatenate([rise, spike, pull_back])
    df = _ohlc(close)

    result = prs.assess_position_risk("XYZ", df, rs_rating=85)

    assert result is not None
    assert result.trend == "uptrend"
    assert result.nearest_resistance is not None
    assert abs(result.nearest_resistance.distance_pct) <= prs.PROXIMITY_THRESHOLD
    assert result.signal == prs.ADD_CANDIDATE
    assert result.score >= 5


def test_signal_is_consistent_with_recommendation_engine_thresholds():
    downtrend = _ohlc(200 - np.arange(260) * 0.3)
    uptrend = _ohlc(100 + np.arange(260) * 0.4)

    exit_result = prs.assess_position_risk("DOWN", downtrend)
    add_result = prs.assess_position_risk("UP", uptrend, rs_rating=90)

    assert exit_result.score <= -3
    assert exit_result.signal == prs.EXIT_WARNING
    assert add_result.score >= 5
    assert add_result.signal == prs.ADD_CANDIDATE


def test_rs_rating_is_passed_through_unchanged():
    close = 100 + np.arange(260) * 0.4
    df = _ohlc(close)
    result = prs.assess_position_risk("XYZ", df, rs_rating=88)
    assert result.rs_rating == 88


def test_get_portfolio_positions_risk_skips_tickers_with_no_data():
    class _StubMarketData:
        def get_bulk_ohlcv(self, tickers, start, end):
            close = 100 + np.arange(260) * 0.4
            return {"AAPL": _ohlc(close)}  # "MISSING" deliberately absent

    results = prs.get_portfolio_positions_risk(["AAPL", "MISSING"], _StubMarketData())

    assert [r.ticker for r in results] == ["AAPL"]


def test_get_portfolio_positions_risk_empty_tickers_returns_empty():
    class _StubMarketData:
        def get_bulk_ohlcv(self, tickers, start, end):
            raise AssertionError("should not be called for an empty ticker list")

    assert prs.get_portfolio_positions_risk([], _StubMarketData()) == []
