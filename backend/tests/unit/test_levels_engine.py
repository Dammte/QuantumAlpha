from datetime import date

import pytest

from app.services import levels_engine as le
from app.services import trade_geometry as tg
from app.services.technical_analysis import PriceLevel, Stage, TrendState

# Sexta auditoría (texto literal completo, Parte 6.2): el gate son 5
# criterios eliminatorios - liquidity_ok, data_quality_ok, weekly_not_stage4,
# no_fast_bearish_cross, no_event_risk - ya no los 6 de la reconstrucción
# anterior (tendencia/parabólico/sobrecompra/OBV/par rápido/R:R). El R:R
# mínimo se separó hacia la viabilidad del disparador (trade_geometry.py),
# nunca fue un criterio del gate de elegibilidad en el texto literal.
#
# A support at exactly the 3% pullback-proximity boundary and no resistance -
# relies on trade_geometry's fixed 2:1 fallback target (no resistance to
# second-guess it), so risk_reward is deterministically 2.0 regardless of the
# exact stop math.


def _passing_kwargs() -> dict:
    return dict(
        price=100.0,
        trend=TrendState.UPTREND,
        atr14=2.0,
        nearest_support=PriceLevel(price=97.0, kind="support", strength=2, distance_pct=-0.03),
        nearest_resistance=None,
        weekly_stage=Stage.STAGE_2,
        liquidity_ok=True,
        fast_pair_bearish_signal=None,
        next_earnings_date=None,
        as_of=date(2026, 9, 15),
    )


def test_all_conditions_pass_is_a_clean_gate():
    result = le.evaluate_gate(**_passing_kwargs())
    assert result.passes is True
    assert all(c.passed for c in result.conditions)
    assert result.eligibility.passes is True
    assert result.entry_trigger is not None
    assert result.entry_trigger.trigger_type == "pullback_bounce"
    assert result.stop_and_target.risk_reward == pytest.approx(2.0)


def test_entry_geometry_is_none_without_ema21_ema55():
    # Parte 7: callers that don't have real EMA reads (e.g. replay_gate_at's
    # point-in-time backtest replay) simply don't get the richer geometry -
    # never a guess built from stop_and_target's simpler numbers.
    result = le.evaluate_gate(**_passing_kwargs())
    assert result.entry_geometry is None


def test_entry_geometry_computed_when_ema21_ema55_are_given():
    result = le.evaluate_gate(**_passing_kwargs(), ema21=98.0, ema55=90.0)
    assert result.entry_geometry is not None
    assert result.entry_geometry.viable is True
    assert result.entry_geometry.entry_type == tg.EntryType.PULLBACK_SUPPORT
    # No capital passed in at this layer - evaluate_gate never sizes.
    assert result.entry_geometry.shares_for_risk_budget is None


def test_gate_fails_on_insufficient_liquidity():
    result = le.evaluate_gate(**{**_passing_kwargs(), "liquidity_ok": False})
    assert result.passes is False
    assert result.eligibility.liquidity_ok is False
    condition = next(c for c in result.conditions if "Liquidez" in c.label)
    assert condition.passed is False


def test_gate_fails_on_insufficient_data_quality():
    result = le.evaluate_gate(**{**_passing_kwargs(), "data_quality_ok": False})
    assert result.passes is False
    assert result.eligibility.data_quality_ok is False
    condition = next(c for c in result.conditions if "Datos suficientes" in c.label)
    assert condition.passed is False


def test_data_quality_ok_defaults_to_true():
    # El propio llamador ya garantiza MIN_BARS_REQUIRED antes de construir
    # cualquier lectura - en la práctica, casi siempre True.
    result = le.evaluate_gate(**_passing_kwargs())
    assert result.eligibility.data_quality_ok is True


def test_gate_fails_when_weekly_stage_is_stage4():
    result = le.evaluate_gate(**{**_passing_kwargs(), "weekly_stage": Stage.STAGE_4})
    assert result.passes is False
    assert result.eligibility.weekly_not_stage4 is False
    condition = next(c for c in result.conditions if "Fase 4" in c.label)
    assert condition.passed is False


def test_gate_fails_when_weekly_stage_is_unknown():
    # Parte 6.2, literal: "unknown NO pasa" - un historial semanal
    # insuficiente (< ~3,85 años) no puede rescatarse asumiendo que no está
    # en Fase 4.
    result = le.evaluate_gate(**{**_passing_kwargs(), "weekly_stage": None})
    assert result.passes is False
    assert result.eligibility.weekly_not_stage4 is False


def test_gate_passes_on_any_weekly_stage_other_than_4():
    for stage in (Stage.STAGE_1, Stage.STAGE_2, Stage.STAGE_3):
        result = le.evaluate_gate(**{**_passing_kwargs(), "weekly_stage": stage})
        assert result.eligibility.weekly_not_stage4 is True


def test_gate_fails_on_fast_pair_bearish_veto():
    result = le.evaluate_gate(**{**_passing_kwargs(), "fast_pair_bearish_signal": "death cross EMA21/55"})
    assert result.passes is False
    assert result.eligibility.no_fast_bearish_cross is False
    condition = next(c for c in result.conditions if "par rápido" in c.label)
    assert condition.passed is False


def test_gate_fails_when_earnings_are_within_the_event_risk_window():
    as_of = date(2026, 9, 15)
    result = le.evaluate_gate(
        **{**_passing_kwargs(), "as_of": as_of, "next_earnings_date": date(2026, 9, 20)}
    )
    assert result.passes is False
    assert result.eligibility.no_event_risk is False
    condition = next(c for c in result.conditions if "riesgo de evento" in c.label)
    assert condition.passed is False


def test_gate_passes_when_earnings_are_beyond_the_event_risk_window():
    as_of = date(2026, 9, 15)
    result = le.evaluate_gate(
        **{**_passing_kwargs(), "as_of": as_of, "next_earnings_date": date(2026, 10, 15)}
    )
    assert result.eligibility.no_event_risk is True


def test_gate_treats_no_known_earnings_date_as_no_event_risk():
    # None significa "no se conoce ninguna fecha próxima" - una comprobación
    # exitosa que no encontró nada, no una comprobación fallida (ver el
    # docstring de `_no_event_risk`). Cuenta como sin riesgo, no como "no se
    # pudo comprobar".
    result = le.evaluate_gate(**{**_passing_kwargs(), "next_earnings_date": None})
    assert result.eligibility.no_event_risk is True


def test_gate_treats_a_past_earnings_date_as_no_event_risk():
    result = le.evaluate_gate(
        **{**_passing_kwargs(), "as_of": date(2026, 9, 15), "next_earnings_date": date(2026, 8, 1)}
    )
    assert result.eligibility.no_event_risk is True


def test_gate_result_lists_every_condition_even_when_several_fail_at_once():
    kwargs = {**_passing_kwargs(), "liquidity_ok": False, "weekly_stage": Stage.STAGE_4}
    result = le.evaluate_gate(**kwargs)
    assert len(result.conditions) == 5
    failed_labels = {c.label for c in result.conditions if not c.passed}
    assert any("Liquidez" in label for label in failed_labels)
    assert any("Fase 4" in label for label in failed_labels)


def test_eligibility_failing_lists_only_the_failed_criteria():
    result = le.evaluate_gate(**{**_passing_kwargs(), "liquidity_ok": False})
    assert len(result.eligibility.failing) == 1
    assert "Liquidez" in result.eligibility.failing[0]


def test_entry_trigger_none_does_not_by_itself_fail_the_gate():
    # A clean setup with no support/resistance close enough to define a
    # watchable level yet is still a passing gate - "no imminent trigger"
    # and "bad entry" are different questions (see module docstring).
    result = le.evaluate_gate(**{**_passing_kwargs(), "nearest_support": None, "nearest_resistance": None})
    assert result.entry_trigger is None
    assert result.passes is True


def test_a_poor_reward_risk_no_longer_fails_the_gate_itself():
    # Parte 5.2/7.4: el R:R mínimo es viabilidad del disparador
    # (`trade_geometry.compute_entry_geometry`), no un criterio de
    # elegibilidad del gate - un ticker puede aprobar el gate sin tener hoy
    # una entrada geométricamente viable.
    kwargs = {
        **_passing_kwargs(),
        "nearest_resistance": PriceLevel(price=104.5, kind="resistance", strength=1, distance_pct=0.045),
    }
    result = le.evaluate_gate(**kwargs)
    assert result.stop_and_target.risk_reward < 1.5
    assert result.passes is True
