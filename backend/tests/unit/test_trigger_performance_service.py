from datetime import UTC, datetime

import pandas as pd
import pytest

from app.domain.models.trigger_event import TriggerEvent
from app.services import trigger_performance_service as tps


def _close_series(start: str, closes: list[float]) -> pd.Series:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.Series(closes, index=dates, dtype=float)


def _event(
    ticker: str,
    event_type: str,
    when: datetime,
    entity_type: str = "ticker",
    price: float = 100.0,
) -> TriggerEvent:
    return TriggerEvent(
        id=None,
        entity_type=entity_type,
        entity_key=ticker,
        event_type=event_type,
        previous_value=None,
        new_value=None,
        occurred_at=when,
        details={"price": price},
    )


# --- compute_trigger_outcomes -------------------------------------------------


def test_compute_trigger_outcomes_measures_gate_passed_hit_rate():
    close = _close_series("2024-01-01", [100.0] * 10 + [110.0] * 60)  # +10% by bar 10, flat after
    when = datetime.combine(close.index[0].date(), datetime.min.time(), tzinfo=UTC)
    events = [_event("AAPL", "gate_passed", when)]

    outcomes = tps.compute_trigger_outcomes(events, {"AAPL": close})

    by_horizon = {o.horizon_days: o for o in outcomes if o.event_type == "gate_passed"}
    assert by_horizon[10].n == 1
    assert by_horizon[10].hit_rate == 1.0
    assert by_horizon[10].mean_return == pytest.approx(0.10)


def test_compute_trigger_outcomes_measures_entry_triggered_separately_from_gate_passed():
    close = _close_series("2024-01-01", list(100.0 + i for i in range(80)))
    when = datetime.combine(close.index[0].date(), datetime.min.time(), tzinfo=UTC)
    events = [
        _event("AAPL", "gate_passed", when),
        _event("AAPL", "entry_triggered", when),
    ]

    outcomes = tps.compute_trigger_outcomes(events, {"AAPL": close})

    event_types = {o.event_type for o in outcomes}
    assert event_types == {"gate_passed", "entry_triggered"}


def test_compute_trigger_outcomes_ignores_exit_urgency_changed():
    close = _close_series("2024-01-01", [100.0] * 80)
    when = datetime.combine(close.index[0].date(), datetime.min.time(), tzinfo=UTC)
    events = [_event("AAPL", "exit_urgency_changed", when, entity_type="position")]

    outcomes = tps.compute_trigger_outcomes(events, {"AAPL": close})

    assert outcomes == []


def test_compute_trigger_outcomes_ignores_gate_failed():
    # Only gate_passed/entry_triggered are entry-side hypotheses worth
    # measuring - gate_failed is the absence of a call, not a call itself.
    close = _close_series("2024-01-01", [100.0] * 80)
    when = datetime.combine(close.index[0].date(), datetime.min.time(), tzinfo=UTC)
    events = [_event("AAPL", "gate_failed", when)]

    outcomes = tps.compute_trigger_outcomes(events, {"AAPL": close})

    assert outcomes == []


def test_compute_trigger_outcomes_skips_tickers_with_no_price_data():
    when = datetime(2024, 1, 2, tzinfo=UTC)
    events = [_event("UNKNOWN", "gate_passed", when)]

    outcomes = tps.compute_trigger_outcomes(events, {})

    assert outcomes == []


def test_compute_trigger_outcomes_negative_return_gives_zero_hit_rate():
    close = _close_series("2024-01-01", [100.0] * 10 + [90.0] * 60)  # -10% by bar 10
    when = datetime.combine(close.index[0].date(), datetime.min.time(), tzinfo=UTC)
    events = [_event("AAPL", "gate_passed", when)]

    outcomes = tps.compute_trigger_outcomes(events, {"AAPL": close})

    by_horizon = {o.horizon_days: o for o in outcomes}
    assert by_horizon[10].hit_rate == 0.0
    assert by_horizon[10].mean_return == pytest.approx(-0.10)


# --- build_trigger_performance_report -----------------------------------------


class _FakeMarketData:
    def __init__(self, price_by_ticker: dict[str, pd.Series]) -> None:
        self._price_by_ticker = price_by_ticker

    def get_bulk_ohlcv(self, tickers, start, end):
        return {
            t: pd.DataFrame({"close": s})
            for t, s in self._price_by_ticker.items()
            if t in tickers
        }


def test_build_trigger_performance_report_returns_empty_when_no_measured_events():
    report = tps.build_trigger_performance_report([], _FakeMarketData({}))
    assert report.outcomes == []


def test_build_trigger_performance_report_fetches_only_measured_tickers():
    close = _close_series("2024-01-01", [100.0] * 10 + [105.0] * 60)
    when = datetime.combine(close.index[0].date(), datetime.min.time(), tzinfo=UTC)
    events = [
        _event("AAPL", "gate_passed", when),
        _event("MSFT", "exit_urgency_changed", when, entity_type="position"),
    ]

    report = tps.build_trigger_performance_report(events, _FakeMarketData({"AAPL": close}))

    assert any(o.event_type == "gate_passed" for o in report.outcomes)
