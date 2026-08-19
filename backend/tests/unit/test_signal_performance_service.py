from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from app.domain.models.position_signal_snapshot import PositionSignalSnapshot
from app.domain.models.recommendation_snapshot import RecommendationSnapshot
from app.services import signal_performance_service as sps


def _close_series(start: str, closes: list[float]) -> pd.Series:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.Series(closes, index=dates, dtype=float)


def _rec_snapshot(ticker: str, verdict: str, when: datetime, score: int = 5, price: float = 100.0):
    return RecommendationSnapshot(ticker=ticker, created_at=when, verdict=verdict, score=score, price=price)


def _pos_snapshot(
    ticker: str,
    signal: str,
    when: datetime,
    portfolio_id: int = 1,
    price: float = 100.0,
    exit_urgency: str | None = None,
    score: int = 0,
    r_multiple: float | None = None,
):
    return PositionSignalSnapshot(
        portfolio_id=portfolio_id, ticker=ticker, created_at=when, signal=signal, exit_urgency=exit_urgency,
        score=score, price=price, r_multiple=r_multiple, engine_version="v1",
    )


# --- forward_return ----------------------------------------------------------


def test_forward_return_basic():
    close = _close_series("2024-01-01", [100.0] * 20 + [110.0] * 20)  # jumps to 110 at bar 20
    result = sps.forward_return(close, snapshot_date=close.index[10].date(), horizon_days=10)
    # bar 10 is 100.0, bar 20 (10 sessions later) is 110.0 -> +10%
    assert result == pytest.approx(0.10)


def test_forward_return_none_when_horizon_extends_past_available_data():
    close = _close_series("2024-01-01", [100.0] * 10)
    result = sps.forward_return(close, snapshot_date=close.index[5].date(), horizon_days=10)
    assert result is None


def test_forward_return_none_when_snapshot_date_is_after_all_data():
    close = _close_series("2024-01-01", [100.0] * 10)
    result = sps.forward_return(close, snapshot_date=date(2030, 1, 1), horizon_days=5)
    assert result is None


def test_forward_return_snaps_to_the_next_trading_day_for_a_weekend_snapshot():
    close = _close_series("2024-01-01", list(100.0 + np.arange(20)))  # Mon 2024-01-01 is a Monday
    # 2024-01-06 is a Saturday - not in the index at all.
    result = sps.forward_return(close, snapshot_date=date(2024, 1, 6), horizon_days=1)
    assert result is not None


def test_forward_return_none_on_empty_series():
    assert sps.forward_return(pd.Series(dtype=float), date(2024, 1, 1), 5) is None


# --- compute_verdict_outcomes --------------------------------------------------


def test_verdict_outcomes_aggregates_across_snapshots():
    # FORWARD_HORIZONS = (5, 10, 21, 63): a jump at bar 21 relative to bar 0
    # means every horizon <= 21 sees the full +10% move, and 63 has no data
    # yet (only 40 bars total) so it's simply absent, not fabricated.
    close = _close_series("2024-01-01", [100.0] * 21 + [110.0] * 19)
    price_by_ticker = {"AAPL": close}
    snapshots = [
        _rec_snapshot("AAPL", "comprar", datetime.combine(close.index[0].date(), datetime.min.time())),
    ]
    outcomes = sps.compute_verdict_outcomes(snapshots, price_by_ticker)
    by_horizon = {o.horizon_days: o for o in outcomes if o.label == "comprar"}
    assert 63 not in by_horizon  # not enough history yet - never fabricated
    assert by_horizon[21].n == 1
    assert by_horizon[21].mean_return == pytest.approx(0.10)


def test_verdict_outcomes_skips_tickers_with_no_price_data():
    snapshots = [_rec_snapshot("UNKNOWN", "comprar", datetime(2024, 1, 1))]
    outcomes = sps.compute_verdict_outcomes(snapshots, price_by_ticker={})
    assert outcomes == []


def test_verdict_outcomes_hit_rate_and_mean_return_are_correct():
    # Two "comprar" snapshots at bar 0: one ticker rises +10% by bar 21, the
    # other falls -10% - hit_rate at horizon=21 should read exactly 0.5 and
    # the two moves should average out to a 0 mean return.
    dates = pd.bdate_range("2024-01-01", periods=40)
    up = pd.Series([100.0] * 21 + [110.0] * 19, index=dates)
    down = pd.Series([100.0] * 21 + [90.0] * 19, index=dates)
    price_by_ticker = {"UP": up, "DOWN": down}
    snapshots = [
        _rec_snapshot("UP", "comprar", datetime.combine(dates[0].date(), datetime.min.time())),
        _rec_snapshot("DOWN", "comprar", datetime.combine(dates[0].date(), datetime.min.time())),
    ]
    outcomes = sps.compute_verdict_outcomes(snapshots, price_by_ticker)
    at_21 = next(o for o in outcomes if o.horizon_days == 21)
    assert at_21.n == 2
    assert at_21.hit_rate == pytest.approx(0.5)
    assert at_21.mean_return == pytest.approx(0.0, abs=1e-9)  # +10% and -10% average to 0


# --- compute_signal_outcomes ---------------------------------------------------


def test_signal_outcomes_groups_by_signal_not_verdict():
    close = _close_series("2024-01-01", [100.0] * 30 + [90.0] * 30)  # -10% after bar 30
    snapshots = [_pos_snapshot("AAPL", "hold", datetime.combine(close.index[0].date(), datetime.min.time()))]
    outcomes = sps.compute_signal_outcomes(snapshots, {"AAPL": close})
    assert all(o.label == "hold" for o in outcomes)
    assert any(o.n == 1 for o in outcomes)


# --- find_false_negatives -------------------------------------------------------


def test_false_negative_caught_when_hold_precedes_a_real_drop():
    # FALSE_NEGATIVE_HORIZON_DAYS = 10: the drop must land by bar 10.
    close = _close_series("2024-01-01", [100.0] * 10 + [93.0] * 20)  # -7% by bar 10 (>5% drop)
    snap = _pos_snapshot("AAPL", "hold", datetime.combine(close.index[0].date(), datetime.min.time()), price=100.0)
    negatives = sps.find_false_negatives([snap], {"AAPL": close})
    assert len(negatives) == 1
    assert negatives[0].ticker == "AAPL"
    assert negatives[0].return_pct < -0.05


def test_no_false_negative_when_drop_is_below_the_5pct_threshold():
    close = _close_series("2024-01-01", [100.0] * 10 + [97.0] * 20)  # only -3%
    snap = _pos_snapshot("AAPL", "hold", datetime.combine(close.index[0].date(), datetime.min.time()))
    assert sps.find_false_negatives([snap], {"AAPL": close}) == []


def test_no_false_negative_for_non_hold_signals_even_with_a_big_drop():
    close = _close_series("2024-01-01", [100.0] * 10 + [80.0] * 20)  # -20%
    snap = _pos_snapshot(
        "AAPL", "exit_warning", datetime.combine(close.index[0].date(), datetime.min.time())
    )
    assert sps.find_false_negatives([snap], {"AAPL": close}) == []


def test_false_negative_price_after_matches_the_actual_forward_price():
    close = _close_series("2024-01-01", [100.0] * 10 + [90.0] * 20)  # exactly -10% by bar 10
    snap = _pos_snapshot("AAPL", "hold", datetime.combine(close.index[0].date(), datetime.min.time()), price=100.0)
    negatives = sps.find_false_negatives([snap], {"AAPL": close})
    assert negatives[0].price_after == pytest.approx(90.0)
    assert negatives[0].return_pct == pytest.approx(-0.10)


# --- build_signal_performance_report (orchestration) ----------------------------


class _StubMarketData:
    def __init__(self, close_by_ticker: dict[str, pd.Series]):
        self._close_by_ticker = close_by_ticker
        self.call_count = 0
        self.requested_tickers: list[str] = []

    def get_bulk_ohlcv(self, tickers, start, end):
        self.call_count += 1
        self.requested_tickers.extend(tickers)
        return {
            t: pd.DataFrame({"close": s, "open": s, "high": s, "low": s, "volume": 1.0})
            for t, s in self._close_by_ticker.items()
            if t in tickers
        }


def test_build_report_fetches_every_distinct_ticker_in_a_single_batched_call():
    close = _close_series("2024-01-01", [100.0] * 40)
    market_data = _StubMarketData({"AAPL": close, "MSFT": close})
    rec_snapshots = [_rec_snapshot("AAPL", "comprar", datetime(2024, 1, 2))]
    pos_snapshots = [_pos_snapshot("MSFT", "hold", datetime(2024, 1, 2))]

    report = sps.build_signal_performance_report(rec_snapshots, pos_snapshots, market_data)

    assert market_data.call_count == 1  # never one call per ticker
    assert set(market_data.requested_tickers) == {"AAPL", "MSFT"}
    assert isinstance(report, sps.SignalPerformanceReport)


def test_build_report_empty_snapshots_returns_empty_report_without_calling_market_data():
    class _NoCallMarketData:
        def get_bulk_ohlcv(self, tickers, start, end):
            raise AssertionError("should not be called with no snapshots")

    report = sps.build_signal_performance_report([], [], _NoCallMarketData())
    assert report.verdict_outcomes == []
    assert report.signal_outcomes == []
    assert report.false_negatives == []
