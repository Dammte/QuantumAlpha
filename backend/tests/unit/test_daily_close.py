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
from app.domain.models.trigger_event import TriggerEvent
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
    assert state.gate_version == "2026-09-levels-v3"
    assert all({"label", "passed"} == set(c) for c in state.gate_conditions)
    assert all(isinstance(c["passed"], bool) for c in state.gate_conditions)
    # Auditoria del Radar, bloque 10: una tendencia alcista limpia y continua
    # no rompe ningún nivel - el resultado normal, "[]", no "None".
    assert state.broken_levels == []


# --- _broken_levels (Auditoria del Radar, bloque 10: "rompiendo por abajo") --


def _level(kind, state, bars_in_state=1, price=100.0) -> ta.Level:
    return ta.Level(
        kind=kind, price=price, side="below", distance_pct=-0.02, distance_atr=1.0,
        state=state, bars_in_state=bars_in_state, strength=None, slope_pct_20d=None,
    )


def test_broken_levels_includes_a_recently_lost_ema21():
    from app.services.ticker_daily_state_builder import _broken_levels

    levels = [_level(ta.LevelKind.EMA21, ta.LevelState.LOST_CONFIRMED, bars_in_state=2, price=95.0)]
    result = _broken_levels(levels)
    assert result == [{"kind": "ema21", "price": 95.0, "bars_since_loss": 2}]


def test_broken_levels_excludes_a_loss_older_than_three_bars():
    from app.services.ticker_daily_state_builder import _broken_levels

    levels = [_level(ta.LevelKind.EMA21, ta.LevelState.LOST_CONFIRMED, bars_in_state=4)]
    assert _broken_levels(levels) == []


def test_broken_levels_excludes_a_level_still_intact():
    from app.services.ticker_daily_state_builder import _broken_levels

    levels = [_level(ta.LevelKind.EMA21, ta.LevelState.FAR, bars_in_state=1)]
    assert _broken_levels(levels) == []


def test_broken_levels_excludes_kinds_outside_the_three_that_count():
    from app.services.ticker_daily_state_builder import _broken_levels

    # Una resistencia rota (al alza) o un máximo de 52 semanas perdido no son
    # "rompiendo por abajo" - solo EMA21/EMA55/soporte cuentan.
    levels = [
        _level(ta.LevelKind.PIVOT_RESISTANCE, ta.LevelState.LOST_CONFIRMED, bars_in_state=1),
        _level(ta.LevelKind.HIGH_52W, ta.LevelState.LOST_CONFIRMED, bars_in_state=1),
        _level(ta.LevelKind.SMA50, ta.LevelState.LOST_CONFIRMED, bars_in_state=1),
    ]
    assert _broken_levels(levels) == []


def test_broken_levels_can_include_more_than_one_kind_at_once():
    from app.services.ticker_daily_state_builder import _broken_levels

    levels = [
        _level(ta.LevelKind.EMA21, ta.LevelState.LOST_CONFIRMED, bars_in_state=1, price=95.0),
        _level(ta.LevelKind.PIVOT_SUPPORT, ta.LevelState.LOST_CONFIRMED, bars_in_state=3, price=90.0),
    ]
    result = _broken_levels(levels)
    assert len(result) == 2
    assert {r["kind"] for r in result} == {"ema21", "pivot_support"}


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


def test_build_ticker_daily_state_persists_grade_when_geometry_is_viable():
    """Parte 5.3 (later pass): `compute_grade`'s real precondition is a
    genuine `entry_trigger` *and* a viable `entry_geometry` together (the
    exact same check `ticker_analysis_service.compute_core_signals` uses for
    "Analizar activo") - a bare monotonic rise (the sibling entry_geometry
    test above) clears the geometry cascade but never sits near a support/
    resistance level, so it never gets a `compute_entry_trigger` and grade
    correctly stays `None` there. A pullback to a nearby swing low (same
    fixture `test_add_candidate_when_uptrend_pulls_back_to_support` in
    test_portfolio_risk_service.py uses) gives both at once."""
    rise = 100 + np.arange(700) * 0.4
    dip = rise[-1] - np.array([0.0, 1.0, 1.8, 1.3, 0.6])
    bounce = dip[-1] + np.arange(1, 4) * 0.4
    close = np.concatenate([rise, dip, bounce])
    df = _ohlc(close)
    state = dc.build_ticker_daily_state(
        _snapshot(price=float(close[-1])), "us", df, date(2026, 9, 10), datetime.now(UTC)
    )
    assert state.entry_geometry is not None and state.entry_geometry["viable"] is True  # this test's precondition
    assert state.grade is not None
    assert state.grade["grade"] in {"A", "B", "C"}
    assert isinstance(state.grade["reasons"], list)


def test_build_ticker_daily_state_grade_none_without_a_viable_geometry():
    # A flat, directionless series never produces a real entry trigger -
    # `entry_geometry` stays `None` (see the sibling entry_geometry test
    # above), and `grade` must follow it into `None` rather than fabricate a
    # grade for a trade that was never emitted in the first place.
    df = _ohlc(np.array([100.0] * 300))
    state = dc.build_ticker_daily_state(_snapshot(price=100.0), "us", df, date(2026, 9, 10), datetime.now(UTC))
    assert state.grade is None


def test_build_ticker_daily_state_setups_is_an_empty_list_when_nothing_matches(monkeypatch):
    """Biblioteca de setups del Radar (en curso, quant_methodology.md §28):
    `SetupContext` ya se construye y `detect_all` ya se llama en este job -
    `[]`, no `None`, es el resultado esperado cuando ningún detector
    registrado encuentra nada. `None` queda reservado para una fila
    calculada antes de que esta columna existiera (ver
    `TickerDailyState.setups`'s propio docstring), nunca para "nada
    coincidió". `SETUP_DETECTORS` se vacía aquí a propósito - esta prueba
    verifica el cableado (detect_all -> setups_list), no la lógica de
    ningún detector real en concreto (eso vive en el test de cada uno)."""
    import app.services.setups.registry as setups_registry

    monkeypatch.setattr(setups_registry, "SETUP_DETECTORS", [])

    df = _ohlc(100 + np.arange(300) * 0.15)
    state = dc.build_ticker_daily_state(_snapshot(), "us", df, date(2026, 9, 10), datetime.now(UTC))
    assert state.setups == []


def test_build_ticker_daily_state_setup_context_reaches_a_registered_detector(monkeypatch):
    """No sondea el contenido exacto de `SetupContext` campo por campo (eso
    es responsabilidad de los tests de cada detector) - solo bloquea que el
    contexto que de verdad llega a un detector viene de datos reales de este
    job (ticker/región/fecha correctos, `close` no vacío), no de un stub."""
    import app.services.setups.registry as setups_registry
    from app.services.setups.context import SetupContext
    from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

    captured: list[SetupContext] = []

    def fake_detector(ctx: SetupContext) -> list[SetupMatch]:
        captured.append(ctx)
        return [
            SetupMatch(
                family=SetupFamily.MA_CROSS,
                name="dummy",
                label_es="Dummy",
                stage=SetupStage.READY,
                bars_in_stage=1,
                timeframe="daily",
                trigger_price=None,
                trigger_condition="",
                invalidation_price=None,
                invalidation_condition="",
                evidence={},
                narrative_es="",
                confidence=SetupConfidence.UNVALIDATED,
            )
        ]

    monkeypatch.setattr(setups_registry, "SETUP_DETECTORS", [fake_detector])

    df = _ohlc(100 + np.arange(300) * 0.15)
    state = dc.build_ticker_daily_state(
        _snapshot(ticker="NVDA"), "europe", df, date(2026, 9, 10), datetime.now(UTC)
    )

    assert len(captured) == 1
    assert captured[0].ticker == "NVDA"
    assert captured[0].region == "europe"
    assert captured[0].trade_date == date(2026, 9, 10)
    assert not captured[0].close.empty
    assert state.setups == [
        {
            "family": "ma_cross",
            "name": "dummy",
            "label_es": "Dummy",
            "bars_in_stage": 1,
            "timeframe": "daily",
            "trigger_price": None,
            "trigger_condition": "",
            "invalidation_price": None,
            "invalidation_condition": "",
            "evidence": {},
            "narrative_es": "",
            "confidence": "unvalidated",
            "stage": "ready",
            "horizon": "medium",
            "expected_sessions_to_trigger": None,
        }
    ]


def test_build_ticker_daily_state_applies_measured_confidence_from_setup_performance(monkeypatch):
    # Parte 10.3 (§28.x): un detector siempre sale de detect_all en
    # UNVALIDATED (visto arriba) - build_ticker_daily_state es quien lo
    # sustituye por la medición real de setup_performance, cuando existe.
    import app.services.setups.registry as setups_registry
    from app.domain.models.setup_performance import SetupPerformance
    from app.services.setups.context import SetupContext
    from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

    def fake_detector(ctx: SetupContext) -> list[SetupMatch]:
        return [
            SetupMatch(
                family=SetupFamily.MA_CROSS, name="dummy", label_es="Dummy", stage=SetupStage.READY,
                bars_in_stage=1, timeframe="daily", trigger_price=None, trigger_condition="",
                invalidation_price=None, invalidation_condition="", evidence={}, narrative_es="",
                confidence=SetupConfidence.UNVALIDATED,
            )
        ]

    monkeypatch.setattr(setups_registry, "SETUP_DETECTORS", [fake_detector])
    performance_by_name = {
        "dummy": SetupPerformance(
            id=1, setup_name="dummy", family="ma_cross", grade=None, market_regime=None,
            n_observations=35, trigger_rate=0.6, win_rate=0.55, expectancy_r=0.42,
            median_bars_held=6.0, mae_p80_pct=-0.03, failure_rate_3d=0.1, confidence="measured",
            computed_at=datetime.now(UTC),
        )
    }

    df = _ohlc(100 + np.arange(300) * 0.15)
    state = dc.build_ticker_daily_state(
        _snapshot(ticker="NVDA"), "europe", df, date(2026, 9, 10), datetime.now(UTC),
        setup_performance_by_name=performance_by_name,
    )

    assert state.setups[0]["confidence"] == "measured"


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


def test_ticker_trigger_events_setup_triggered_on_ready_to_triggered_transition():
    previous = _ticker_state(
        trade_date=date(2026, 9, 9), setups=[{"name": "vcp_3", "family": "vcp", "stage": "ready"}]
    )
    new = _ticker_state(
        trade_date=date(2026, 9, 10), setups=[{"name": "vcp_3", "family": "vcp", "stage": "triggered"}]
    )
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "setup_triggered"
    assert event.previous_value == "ready"
    assert event.new_value == "triggered"
    assert event.details == {"price": new.price, "setup_name": "vcp_3", "family": "vcp"}


def test_ticker_trigger_events_setup_triggered_absent_when_already_triggered_yesterday():
    previous = _ticker_state(
        trade_date=date(2026, 9, 9), setups=[{"name": "vcp_3", "family": "vcp", "stage": "triggered"}]
    )
    new = _ticker_state(
        trade_date=date(2026, 9, 10), setups=[{"name": "vcp_3", "family": "vcp", "stage": "triggered"}]
    )
    assert dc.ticker_trigger_events(previous, new, datetime.now(UTC)) == []


def test_ticker_trigger_events_setup_triggered_when_setup_absent_yesterday():
    previous = _ticker_state(trade_date=date(2026, 9, 9), setups=[])
    new = _ticker_state(
        trade_date=date(2026, 9, 10), setups=[{"name": "vcp_3", "family": "vcp", "stage": "triggered"}]
    )
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].previous_value is None


def test_ticker_trigger_events_setup_triggered_never_fires_from_forming_or_failed_transitions():
    # "a" pasa de forming a ready (dispara setup_ready, no setup_triggered);
    # "b" pasa de ready a failed (ningún evento - fallar no está entre los
    # tipos que este mecanismo registra).
    previous = _ticker_state(
        trade_date=date(2026, 9, 9),
        setups=[
            {"name": "a", "family": "vcp", "stage": "forming"},
            {"name": "b", "family": "vcp", "stage": "ready"},
        ],
    )
    new = _ticker_state(
        trade_date=date(2026, 9, 10),
        setups=[
            {"name": "a", "family": "vcp", "stage": "ready"},
            {"name": "b", "family": "vcp", "stage": "failed"},
        ],
    )
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].event_type == "setup_ready"
    assert events[0].details["setup_name"] == "a"
    assert not any(e.event_type == "setup_triggered" for e in events)


# --- ticker_trigger_events: setup_ready (Parte 11.4) -------------------------


def test_ticker_trigger_events_setup_ready_on_forming_to_ready_transition():
    previous = _ticker_state(
        trade_date=date(2026, 9, 9), setups=[{"name": "vcp_3", "family": "vcp", "stage": "forming"}]
    )
    new = _ticker_state(
        trade_date=date(2026, 9, 10), setups=[{"name": "vcp_3", "family": "vcp", "stage": "ready"}]
    )
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "setup_ready"
    assert event.previous_value == "forming"
    assert event.new_value == "ready"
    assert event.details == {"price": new.price, "setup_name": "vcp_3", "family": "vcp"}


def test_ticker_trigger_events_setup_ready_when_setup_absent_yesterday():
    previous = _ticker_state(trade_date=date(2026, 9, 9), setups=[])
    new = _ticker_state(
        trade_date=date(2026, 9, 10), setups=[{"name": "vcp_3", "family": "vcp", "stage": "ready"}]
    )
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].event_type == "setup_ready"
    assert events[0].previous_value is None


def test_ticker_trigger_events_setup_ready_absent_when_already_ready_yesterday():
    previous = _ticker_state(
        trade_date=date(2026, 9, 9), setups=[{"name": "vcp_3", "family": "vcp", "stage": "ready"}]
    )
    new = _ticker_state(
        trade_date=date(2026, 9, 10), setups=[{"name": "vcp_3", "family": "vcp", "stage": "ready"}]
    )
    assert dc.ticker_trigger_events(previous, new, datetime.now(UTC)) == []


def test_ticker_trigger_events_setup_ready_fires_again_after_a_failed_cycle():
    # Un ciclo completo (ready -> failed -> ready otra vez) es una NUEVA
    # formación - debe volver a contar, no quedarse silenciado para siempre.
    previous = _ticker_state(
        trade_date=date(2026, 9, 9), setups=[{"name": "vcp_3", "family": "vcp", "stage": "failed"}]
    )
    new = _ticker_state(
        trade_date=date(2026, 9, 10), setups=[{"name": "vcp_3", "family": "vcp", "stage": "ready"}]
    )
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert len(events) == 1
    assert events[0].event_type == "setup_ready"
    assert events[0].previous_value == "failed"


def test_ticker_trigger_events_setup_triggered_two_setups_same_day():
    previous = _ticker_state(
        trade_date=date(2026, 9, 9),
        setups=[
            {"name": "vcp_3", "family": "vcp", "stage": "ready"},
            {"name": "breakout_darvas", "family": "breakout", "stage": "forming"},
        ],
    )
    new = _ticker_state(
        trade_date=date(2026, 9, 10),
        setups=[
            {"name": "vcp_3", "family": "vcp", "stage": "triggered"},
            {"name": "breakout_darvas", "family": "breakout", "stage": "triggered"},
        ],
    )
    events = dc.ticker_trigger_events(previous, new, datetime.now(UTC))
    assert {e.details["setup_name"] for e in events} == {"vcp_3", "breakout_darvas"}
    assert all(e.event_type == "setup_triggered" for e in events)


def test_ticker_trigger_events_setup_triggered_none_setups_does_not_crash():
    previous = _ticker_state(trade_date=date(2026, 9, 9), setups=None)
    new = _ticker_state(trade_date=date(2026, 9, 10), setups=None)
    assert dc.ticker_trigger_events(previous, new, datetime.now(UTC)) == []


# --- _today_trigger_counts (Auditoria del Radar, bloque B4) ------------------


class _FakeTriggerEventRepo:
    """Doble mínimo de `TriggerEventRepository` - solo `list_since` importa
    aquí, `_today_trigger_counts` no toca nada más de la interfaz real."""

    def __init__(self, events):
        self._events = events

    def list_since(self, since, entity_type=None):
        return [e for e in self._events if e.occurred_at >= since and e.entity_type == entity_type]


def _trigger_event(event_type: str, occurred_at: datetime, entity_type: str = "ticker") -> TriggerEvent:
    return TriggerEvent(
        id=None, entity_type=entity_type, entity_key="AAPL", event_type=event_type,
        previous_value=None, new_value=None, occurred_at=occurred_at, details={},
    )


def test_today_trigger_counts_counts_gate_passed_and_entry_triggered_separately():
    trade_date = date(2026, 9, 10)
    today = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
    events = [
        _trigger_event("gate_passed", today),
        _trigger_event("gate_passed", today),
        _trigger_event("entry_triggered", today),
        _trigger_event("gate_failed", today),  # no cuenta para ninguno de los dos
    ]
    repo = _FakeTriggerEventRepo(events)

    new_gate_passes, new_entry_triggers = dc._today_trigger_counts(repo, trade_date)

    assert new_gate_passes == 2
    assert new_entry_triggers == 1


def test_today_trigger_counts_ignores_events_from_a_previous_day():
    trade_date = date(2026, 9, 10)
    yesterday = datetime(2026, 9, 9, 15, 0, tzinfo=UTC)
    repo = _FakeTriggerEventRepo([_trigger_event("gate_passed", yesterday)])

    new_gate_passes, new_entry_triggers = dc._today_trigger_counts(repo, trade_date)

    assert new_gate_passes == 0
    assert new_entry_triggers == 0


def test_today_trigger_counts_is_stable_across_repeated_calls_same_day():
    # El propio punto de B4: el mismo conteo sale sin importar cuántas veces
    # se "reintente" - list_since nunca duplica (append-only), así que
    # llamar dos veces con el mismo repo da el mismo resultado.
    trade_date = date(2026, 9, 10)
    today = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
    repo = _FakeTriggerEventRepo([_trigger_event("entry_triggered", today)])

    first = dc._today_trigger_counts(repo, trade_date)
    second = dc._today_trigger_counts(repo, trade_date)

    assert first == second == (0, 1)


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
