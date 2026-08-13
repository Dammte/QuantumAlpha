from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.services import portfolio_risk_service as prs
from app.services import recommendation_engine as re
from app.services import technical_analysis as ta


def _ohlc(close: np.ndarray, wiggle: float = 1.0) -> pd.DataFrame:
    close_s = pd.Series(close)
    return pd.DataFrame(
        {"close": close_s, "high": close_s + wiggle, "low": close_s - wiggle, "volume": 1_000_000.0}
    )


def _stub_signals(verdict="esperar", score=0, imminent_cross=None):
    """A minimal stand-in for CoreTickerSignals carrying only what
    assess_position_risk actually reads - used to test its signal/reasons
    logic in isolation from the real indicator pipeline."""
    return SimpleNamespace(
        price=105.0,
        trend=ta.TrendState.SIDEWAYS,
        stage=None,
        ma_cross=None,
        rs_rating=None,
        nearest_support=None,
        nearest_resistance=None,
        recommendation=SimpleNamespace(verdict=verdict, score=score, factors=[]),
        imminent_cross=imminent_cross,
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
    # A perfectly straight-line uptrend (no noise at all) is a pathological
    # fixture for a real indicator set: RSI pins at exactly 100, and the
    # ever-growing distance from a near-flat ATR trips the parabolic-extension
    # and GARCH-high-vol *caution* factors right alongside the bullish
    # trend/stage/RS ones - the engine correctly treating an unrealistically
    # smooth, already-extended move with caution, not a bug. Mild noise around
    # the same slope keeps this a genuine, clean uptrend without that artifact.
    rng = np.random.default_rng(7)
    trend = 100 + np.arange(260) * 0.4
    noise = rng.normal(0, 1.2, 260).cumsum() * 0.15
    uptrend = _ohlc(trend + noise, wiggle=1.5)

    exit_result = prs.assess_position_risk("DOWN", downtrend)
    add_result = prs.assess_position_risk("UP", uptrend, rs_rating=90)

    assert exit_result.score <= re.AVOID_THRESHOLD
    assert exit_result.signal == prs.EXIT_WARNING
    assert add_result.score >= re.BUY_THRESHOLD
    assert add_result.signal == prs.ADD_CANDIDATE


def test_rs_rating_is_passed_through_unchanged():
    close = 100 + np.arange(260) * 0.4
    df = _ohlc(close)
    result = prs.assess_position_risk("XYZ", df, rs_rating=88)
    assert result.rs_rating == 88


def test_hold_escalates_to_watch_when_death_cross_is_imminent(monkeypatch):
    """The concrete answer to "I bought, it rose 10%, then gave it all back in
    2 days": a projected (not yet confirmed) death cross is an earlier
    heads-up than waiting for ma_cross to actually confirm it - a position
    that would otherwise show as a quiet "mantener" gets escalated to
    "vigilar" instead, with the projection spelled out in `reasons`."""
    imminent = ta.ImminentCross(direction="death", bars_until=5, r_squared=0.9)
    monkeypatch.setattr(prs, "compute_core_signals", lambda *a, **k: _stub_signals(imminent_cross=imminent))

    result = prs.assess_position_risk("XYZ", _ohlc(np.array([100.0] * 5)))

    assert result.signal == prs.WATCH
    assert any("cruce de medias bajista" in r for r in result.reasons)


def test_hold_stays_hold_without_an_imminent_cross(monkeypatch):
    monkeypatch.setattr(prs, "compute_core_signals", lambda *a, **k: _stub_signals(imminent_cross=None))
    result = prs.assess_position_risk("XYZ", _ohlc(np.array([100.0] * 5)))
    assert result.signal == prs.HOLD


def test_hold_gets_an_informational_note_when_golden_cross_is_imminent(monkeypatch):
    """An imminent golden cross doesn't itself clear the bar for add_candidate
    (that still requires the full recommendation engine's "comprar" verdict) -
    purely informational for a HOLD, not an escalation."""
    imminent = ta.ImminentCross(direction="golden", bars_until=4, r_squared=0.9)
    monkeypatch.setattr(prs, "compute_core_signals", lambda *a, **k: _stub_signals(imminent_cross=imminent))

    result = prs.assess_position_risk("XYZ", _ohlc(np.array([100.0] * 5)))

    assert result.signal == prs.HOLD
    assert any("cruce de medias alcista" in r for r in result.reasons)


def test_exit_warning_unaffected_by_imminent_cross(monkeypatch):
    """The recommendation engine's own "evitar" verdict already takes priority
    - an imminent-cross projection doesn't need to (and shouldn't) change
    anything about an already-urgent signal."""
    imminent = ta.ImminentCross(direction="golden", bars_until=3, r_squared=0.9)
    stub = _stub_signals(verdict="evitar", score=-5, imminent_cross=imminent)
    monkeypatch.setattr(prs, "compute_core_signals", lambda *a, **k: stub)
    result = prs.assess_position_risk("XYZ", _ohlc(np.array([100.0] * 5)))
    assert result.signal == prs.EXIT_WARNING


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


def test_get_portfolio_positions_risk_isolates_a_ticker_whose_compute_raises(monkeypatch):
    """The exact production bug this test locks in: one holding's GARCH
    optimizer failing to converge, a backtest edge case, or any other
    numerical hiccup on a real ticker's data must never take the rest of the
    portfolio's risk read down with it (previously an uncaught exception here
    propagated straight to a 500 on the whole /risk response)."""

    class _StubMarketData:
        def get_bulk_ohlcv(self, tickers, start, end):
            close = 100 + np.arange(260) * 0.4
            return {t: _ohlc(close) for t in tickers}

    def flaky_assess(ticker, df, benchmark_close=None, rs_rating=None, vix_close=None):
        if ticker == "BAD":
            raise ValueError("simulated GARCH/backtest numerical failure")
        return prs.PositionRisk(
            ticker=ticker,
            currency="USD",
            price=100.0,
            trend="uptrend",
            stage=None,
            ma_cross=None,
            rs_rating=None,
            nearest_support=None,
            nearest_resistance=None,
            signal=prs.HOLD,
            score=0,
            reasons=["ok"],
            signals=None,
        )

    monkeypatch.setattr(prs, "assess_position_risk", flaky_assess)

    results = prs.get_portfolio_positions_risk(["GOOD", "BAD"], _StubMarketData())
    assert [r.ticker for r in results] == ["GOOD"]


class _CountingMarketData:
    """Stub that counts get_bulk_ohlcv calls and how many tickers were asked
    for across all calls, so tests can assert the cache actually avoids
    recomputation instead of just checking the returned values look right."""

    def __init__(self):
        self.call_count = 0
        self.requested_tickers: list[str] = []

    def get_bulk_ohlcv(self, tickers, start, end):
        self.call_count += 1
        self.requested_tickers.extend(tickers)
        close = 100 + np.arange(260) * 0.4
        return {ticker: _ohlc(close) for ticker in tickers}


def test_service_caches_results_across_calls():
    market_data = _CountingMarketData()
    service = prs.PortfolioRiskService()

    first = service.get_positions_risk(["AAPL", "MSFT"], market_data)
    second = service.get_positions_risk(["AAPL", "MSFT"], market_data)

    assert [r.ticker for r in first] == ["AAPL", "MSFT"]
    assert [r.ticker for r in second] == ["AAPL", "MSFT"]
    assert market_data.call_count == 1  # second call served entirely from cache


def test_service_only_recomputes_uncached_tickers():
    market_data = _CountingMarketData()
    service = prs.PortfolioRiskService()

    service.get_positions_risk(["AAPL"], market_data)
    market_data.requested_tickers.clear()  # only care what the *second* call asks for
    service.get_positions_risk(["AAPL", "MSFT"], market_data)

    assert market_data.call_count == 2
    # MSFT (and its benchmark) get (re)fetched; AAPL stayed cached and is never
    # asked for again - asserting non-membership rather than an exact list
    # since the fetch also includes each ticker's benchmark (e.g. ^GSPC).
    assert "MSFT" in market_data.requested_tickers
    assert "AAPL" not in market_data.requested_tickers


def test_service_force_refresh_recomputes_even_when_cached():
    market_data = _CountingMarketData()
    service = prs.PortfolioRiskService()

    service.get_positions_risk(["AAPL"], market_data)
    service.get_positions_risk(["AAPL"], market_data, force_refresh=True)

    assert market_data.call_count == 2


def test_service_preserves_input_ticker_order_regardless_of_thread_completion_order():
    market_data = _CountingMarketData()
    service = prs.PortfolioRiskService()

    tickers = ["MSFT", "AAPL", "NVDA", "GOOG"]
    results = service.get_positions_risk(tickers, market_data)

    assert [r.ticker for r in results] == tickers


def test_service_empty_tickers_returns_empty_without_calling_market_data():
    class _StubMarketData:
        def get_bulk_ohlcv(self, tickers, start, end):
            raise AssertionError("should not be called for an empty ticker list")

    service = prs.PortfolioRiskService()
    assert service.get_positions_risk([], _StubMarketData()) == []


def test_service_isolates_a_ticker_whose_compute_raises(monkeypatch):
    """Same fault-isolation guarantee as
    test_get_portfolio_positions_risk_isolates_a_ticker_whose_compute_raises,
    covering the service's own cache-miss loop (a near-duplicate of the
    module-level function's, since it recomputes independently per ticker)."""
    market_data = _CountingMarketData()

    def flaky_assess(ticker, df, benchmark_close=None, rs_rating=None, vix_close=None):
        if ticker == "BAD":
            raise ValueError("simulated numerical failure")
        return prs.PositionRisk(
            ticker=ticker,
            currency="USD",
            price=100.0,
            trend="uptrend",
            stage=None,
            ma_cross=None,
            rs_rating=None,
            nearest_support=None,
            nearest_resistance=None,
            signal=prs.HOLD,
            score=0,
            reasons=["ok"],
            signals=None,
        )

    monkeypatch.setattr(prs, "assess_position_risk", flaky_assess)

    service = prs.PortfolioRiskService()
    results = service.get_positions_risk(["GOOD", "BAD"], market_data)
    assert [r.ticker for r in results] == ["GOOD"]
