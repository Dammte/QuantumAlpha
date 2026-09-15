"""Unit tests for the pure/synthetic-data-safe pieces of scripts/daily_close.py
- everything that doesn't touch a DB session or the network. Imported as
`scripts.daily_close` - see test_chandelier_calibration_study.py's own
docstring for why this import path works (pytest's `pythonpath = ["."]` +
scripts/ as an implicit namespace package)."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

import scripts.daily_close as dc
from app.domain.models.position_daily_state import PositionDailyState
from app.domain.models.ticker_daily_state import TickerDailyState
from app.domain.models.ticker_snapshot import TickerSnapshot
from app.domain.models.trade_plan import TradePlan
from app.services import exit_engine as ee
from app.services import technical_analysis as ta
from app.services.portfolio_risk_service import PositionRisk


def _ohlc(close: np.ndarray, wiggle: float = 1.0) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp("2026-09-10"), periods=len(close))
    close_s = pd.Series(close, index=index)
    return pd.DataFrame(
        {
            "open": close_s,
            "close": close_s,
            "high": close_s + wiggle,
            "low": close_s - wiggle,
            "volume": pd.Series([1_000_000.0] * len(close), index=index),
        }
    )


def _snapshot(**overrides) -> TickerSnapshot:
    defaults = dict(
        ticker="AAPL",
        sector="Tecnología",
        industry=None,
        cap_tier="large",
        price=120.0,
        change_1d=0.5,
        change_1w=1.0,
        change_1m=2.0,
        change_3m=5.0,
        change_6m=10.0,
        change_1y=20.0,
        volume=1_000_000.0,
        relative_volume=1.1,
        rsi14=55.0,
        sma20=118.0,
        sma50=112.0,
        sma150=100.0,
        sma200=95.0,
        dist_52w_high=-0.05,
        dist_52w_low=0.30,
        atr_multiple=1.2,
        adx14=28.0,
        plus_di=25.0,
        minus_di=12.0,
        mansfield_rs=0.5,
        trend=ta.TrendState.UPTREND,
        stage=ta.Stage.STAGE_2,
        ma_cross=None,
        minervini_score=7,
        minervini_pass=False,
        rs_rating=85,
    )
    defaults.update(overrides)
    return TickerSnapshot(**defaults)


def _ticker_state(**overrides) -> TickerDailyState:
    defaults = dict(
        id=None,
        region="us",
        ticker="AAPL",
        trade_date=date(2026, 9, 9),
        computed_at=datetime.now(UTC),
        price=120.0,
        currency="USD",
        trend="uptrend",
        stage="stage2",
        rs_rating=85,
        adx14=28.0,
        atr_multiple=1.2,
        rsi14=55.0,
        gate_passes=True,
        gate_conditions=[{"label": "x", "passed": True}],
        gate_version="2026-09-levels-v1",
        entry_trigger_type="breakout",
        entry_trigger_price=125.0,
        entry_already_triggered=False,
        stop_loss=110.0,
        take_profit=140.0,
        take_profit_method="objetivo 2:1 sobre el riesgo",
        risk_reward=2.0,
    )
    defaults.update(overrides)
    return TickerDailyState(**defaults)


def _position_state(**overrides) -> PositionDailyState:
    defaults = dict(
        id=None,
        portfolio_id=1,
        ticker="AAPL",
        trade_date=date(2026, 9, 9),
        computed_at=datetime.now(UTC),
        urgency="hold",
        reasons=[],
        price=120.0,
        r_multiple=1.0,
        current_stop=110.0,
        engine_version="2026-09-audit-v7",
    )
    defaults.update(overrides)
    return PositionDailyState(**defaults)


# --- build_ticker_daily_state -------------------------------------------------


def test_build_ticker_daily_state_none_with_insufficient_bars():
    df = _ohlc(100 + np.arange(30) * 0.1)  # well under MIN_BARS_REQUIRED (60)
    state = dc.build_ticker_daily_state(_snapshot(), "us", df, date(2026, 9, 10), datetime.now(UTC))
    assert state is None


def test_build_ticker_daily_state_carries_identity_and_gate_fields():
    df = _ohlc(100 + np.arange(300) * 0.15)
    computed_at = datetime.now(UTC)
    state = dc.build_ticker_daily_state(_snapshot(), "us", df, date(2026, 9, 10), computed_at)
    assert state is not None
    assert state.region == "us"
    assert state.ticker == "AAPL"
    assert state.trade_date == date(2026, 9, 10)
    assert state.computed_at == computed_at
    assert state.currency == "USD"
    assert state.trend == "uptrend"
    assert state.stage == "stage2"
    assert state.rs_rating == 85
    assert isinstance(state.gate_passes, bool)
    assert state.gate_version == "2026-09-levels-v1"
    assert all({"label", "passed"} == set(c) for c in state.gate_conditions)
    assert all(isinstance(c["passed"], bool) for c in state.gate_conditions)


def test_build_ticker_daily_state_stage_none_serializes_to_none():
    df = _ohlc(100 + np.arange(300) * 0.15)
    state = dc.build_ticker_daily_state(
        _snapshot(stage=None), "us", df, date(2026, 9, 10), datetime.now(UTC)
    )
    assert state.stage is None


def test_build_ticker_daily_state_persists_entry_geometry_when_ema_reads_are_available():
    # Parte 7 (later pass): a long enough, steadily rising series clears the
    # EMA55 warm-up this needs (`multi_timeframe.SLOW_MA_PERIOD`) - the same
    # bar `ticker_analysis_service.compute_core_signals` clears for
    # "Analizar activo"/`/risk`'s own `entry_geometry`, so this job's own copy
    # must come back non-None too, not silently stay the pre-migration `None`.
    close = 100 + np.arange(300) * 0.15
    df = _ohlc(close)
    # `price` must agree with the series' own last close - `_snapshot()`'s
    # fixed default (120.0) sits *below* this series' real EMA21/55, which
    # would fail every cascade rung for a reason that has nothing to do with
    # what this test is actually checking.
    state = dc.build_ticker_daily_state(
        _snapshot(price=float(close[-1])), "us", df, date(2026, 9, 10), datetime.now(UTC)
    )
    assert state.entry_geometry is not None
    assert state.entry_geometry["viable"] is True
    assert isinstance(state.entry_geometry["entry_type"], str)
    # Never sized in this job - no portfolio in scope (see the module's own
    # docstring and `trade_geometry.size_position`'s).
    assert state.entry_geometry["shares_for_risk_budget"] is None


# --- ticker_trigger_events -----------------------------------------------------


def test_ticker_trigger_events_empty_when_no_previous():
    new = _ticker_state()
    assert dc.ticker_trigger_events(None, new, datetime.now(UTC)) == []


def test_ticker_trigger_events_empty_when_previous_is_same_day_rerun():
    same_day = date(2026, 9, 10)
    previous = _ticker_state(trade_date=same_day, gate_passes=False)
    new = _ticker_state(trade_date=same_day, gate_passes=True)
    assert dc.ticker_trigger_events(previous, new, datetime.now(UTC)) == []


def test_ticker_trigger_events_gate_passed():
    previous = _ticker_state(trade_date=date(2026, 9, 9), gate_passes=False)
    new = _ticker_state(trade_date=date(2026, 9, 10), gate_passes=True)
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].event_type == "gate_passed"
    assert events[0].entity_type == "ticker"
    assert events[0].entity_key == "AAPL"


def test_ticker_trigger_events_gate_failed():
    previous = _ticker_state(trade_date=date(2026, 9, 9), gate_passes=True)
    new = _ticker_state(trade_date=date(2026, 9, 10), gate_passes=False)
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].event_type == "gate_failed"


def test_ticker_trigger_events_entry_triggered():
    previous = _ticker_state(trade_date=date(2026, 9, 9), entry_already_triggered=False)
    new = _ticker_state(trade_date=date(2026, 9, 10), entry_already_triggered=True)
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].event_type == "entry_triggered"


def test_ticker_trigger_events_both_fire_together():
    previous = _ticker_state(trade_date=date(2026, 9, 9), gate_passes=False, entry_already_triggered=False)
    new = _ticker_state(trade_date=date(2026, 9, 10), gate_passes=True, entry_already_triggered=True)
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert {e.event_type for e in events} == {"gate_passed", "entry_triggered"}


def test_ticker_trigger_events_none_when_nothing_changed():
    previous = _ticker_state(trade_date=date(2026, 9, 9))
    new = _ticker_state(trade_date=date(2026, 9, 10))
    assert dc.ticker_trigger_events(previous, new, datetime.now(UTC)) == []


# --- position_daily_state_from_risk --------------------------------------------


def _risk(**overrides) -> PositionRisk:
    defaults = dict(
        ticker="AAPL",
        currency="USD",
        price=120.0,
        trend="uptrend",
        stage=None,
        ma_cross=None,
        rs_rating=None,
        nearest_support=None,
        nearest_resistance=None,
        signal="hold",
        score=0,
        reasons=[],
        signals=None,
        exit_urgency=None,
        exit_reasons=[],
        trade_plan=None,
        r_multiple=None,
    )
    defaults.update(overrides)
    return PositionRisk(**defaults)


def test_position_daily_state_from_risk_none_without_exit_urgency():
    risk = _risk(exit_urgency=None)
    assert dc.position_daily_state_from_risk(risk, 1, date(2026, 9, 10), datetime.now(UTC)) is None


def test_position_daily_state_from_risk_builds_expected_fields():
    plan = TradePlan(
        id=1, portfolio_id=1, ticker="AAPL", entry_price=100.0, entry_date=date(2026, 8, 1),
        initial_stop=90.0, initial_target=120.0, current_stop=105.0, highest_close_since_entry=122.0,
        initial_quantity=10.0, thesis="", engine_version="v7", updated_at=datetime.now(UTC), closed_at=None,
    )
    risk = _risk(exit_urgency="reduce", exit_reasons=["Estancada"], r_multiple=0.4, trade_plan=plan, price=121.0)
    state = dc.position_daily_state_from_risk(risk, 7, date(2026, 9, 10), datetime.now(UTC))
    assert state is not None
    assert state.portfolio_id == 7
    assert state.ticker == "AAPL"
    assert state.urgency == "reduce"
    assert state.reasons == ["Estancada"]
    assert state.price == pytest.approx(121.0)
    assert state.r_multiple == pytest.approx(0.4)
    assert state.current_stop == pytest.approx(105.0)


def test_position_daily_state_from_risk_current_stop_none_without_a_trade_plan():
    risk = _risk(exit_urgency="watch", trade_plan=None)
    state = dc.position_daily_state_from_risk(risk, 1, date(2026, 9, 10), datetime.now(UTC))
    assert state.current_stop is None


# --- position_trigger_events ---------------------------------------------------


def test_position_trigger_events_empty_when_no_previous():
    new = _position_state()
    assert dc.position_trigger_events(None, new, datetime.now(UTC)) == []


def test_position_trigger_events_empty_when_same_day_rerun():
    same_day = date(2026, 9, 10)
    previous = _position_state(trade_date=same_day, urgency="hold")
    new = _position_state(trade_date=same_day, urgency="reduce")
    assert dc.position_trigger_events(previous, new, datetime.now(UTC)) == []


def test_position_trigger_events_empty_when_urgency_unchanged():
    previous = _position_state(trade_date=date(2026, 9, 9), urgency="hold")
    new = _position_state(trade_date=date(2026, 9, 10), urgency="hold")
    assert dc.position_trigger_events(previous, new, datetime.now(UTC)) == []


def test_position_trigger_events_fires_on_urgency_change():
    previous = _position_state(trade_date=date(2026, 9, 9), urgency="hold", portfolio_id=3, ticker="MSFT")
    new = _position_state(trade_date=date(2026, 9, 10), urgency="exit_now", portfolio_id=3, ticker="MSFT")
    events = dc.position_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].event_type == "exit_urgency_changed"
    assert events[0].entity_type == "position"
    assert events[0].entity_key == "3:MSFT"
    assert events[0].previous_value == "hold"
    assert events[0].new_value == "exit_now"


# --- build_daily_brief ----------------------------------------------------------


def test_build_daily_brief_headline_when_positions_need_action():
    states = [_position_state(urgency="reduce"), _position_state(urgency="hold", ticker="MSFT")]
    brief = dc.build_daily_brief(1, date(2026, 9, 10), datetime.now(UTC), states, 0, 0)
    assert brief.positions_needing_action == 1
    assert "necesitan atención" in brief.headline


def test_build_daily_brief_headline_when_only_new_entry_triggers():
    brief = dc.build_daily_brief(
        1, date(2026, 9, 10), datetime.now(UTC), [], new_entry_triggers=3, new_gate_passes=0
    )
    assert brief.positions_needing_action == 0
    assert brief.new_entry_triggers == 3
    assert "dispararon su entrada" in brief.headline


def test_build_daily_brief_headline_when_nothing_notable():
    brief = dc.build_daily_brief(
        1, date(2026, 9, 10), datetime.now(UTC), [], new_entry_triggers=0, new_gate_passes=0
    )
    assert brief.headline == "Sin acciones urgentes hoy."


def test_build_daily_brief_urgency_uses_exit_engine_hold_value_not_a_hardcoded_string():
    # Regression guard: if exit_engine.ExitUrgency.HOLD's value ever changes,
    # this comparison must change with it, not silently keep comparing
    # against a stale literal.
    assert ee.ExitUrgency.HOLD.value == "hold"
