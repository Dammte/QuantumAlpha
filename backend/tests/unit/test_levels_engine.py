import inspect
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


# --- Parte 5.3: grados A/B/C ---------------------------------------------


def _trigger(trigger_price: float = 100.0) -> tg.EntryTrigger:
    return tg.EntryTrigger(trigger_type="breakout", trigger_price=trigger_price, already_triggered=True)


def _geometry(
    risk_atr: float = 1.0, risk_reward_net: float = 3.0, viable: bool = True, rejection_reason: str | None = None
) -> tg.TradeGeometry:
    return tg.TradeGeometry(
        entry_price=100.0, stop_price=100.0 - risk_atr * 2.0, stop_basis="bajo el soporte",
        entry_type=tg.EntryType.PULLBACK_SUPPORT, risk_pct=0.02, risk_atr=risk_atr, risk_ceiling_pct=0.05,
        target_price=110.0, target_basis="objetivo 2:1 sobre el riesgo", reward_pct=0.10,
        risk_reward_gross=risk_reward_net, risk_reward_net=risk_reward_net,
        shares_for_risk_budget=None, position_value=None, pct_of_portfolio=None,
        viable=viable, rejection_reason=rejection_reason,
    )


def test_grade_a_when_every_condition_clears_its_bar():
    # Precio ya en el nivel (distancia 0 ATR), riesgo 1.0 ATR, R:R neto 3.0,
    # volumen relativo 1.5x, semanal alcista - todo dentro de los umbrales A.
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5,
    )
    assert result.grade == le.Grade.A


def test_grade_b_when_it_clears_b_but_not_a():
    # Distancia 0, riesgo 1.0 ATR, R:R 3.0 - pero volumen relativo bajo, así
    # que el grado A (que lo exige >= 1.2) no se alcanza; B no lo exige.
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=0.8,
    )
    assert result.grade == le.Grade.B


def test_grade_c_when_it_only_clears_the_viability_floor():
    # Riesgo 1.9 ATR (por encima del techo de B, 2.0 está bien pero R:R bajo
    # rompe B) - viable igualmente (ya lo garantiza trade_geometry), C.
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0),
        geometry=_geometry(risk_atr=1.9, risk_reward_net=1.6),
        weekly_bullish=False,
    )
    assert result.grade == le.Grade.C


def test_grade_none_when_geometry_is_not_viable():
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0),
        geometry=_geometry(viable=False, rejection_reason="posición demasiado pequeña"),
        weekly_bullish=True,
    )
    assert result.grade is None
    assert "posición demasiado pequeña" in result.reasons[0]
    assert result.distance_atr is None


def test_grade_result_exposes_distance_atr_for_the_radar_sort_key():
    # Parte 9.1 (biblioteca de setups del Radar): la clave de ordenación
    # necesita "distancia al gatillo en ATR" sin volver a calcularla -
    # precio a 102, gatillo en 100, ATR 2.0 -> 1.0 ATR de distancia.
    result = le.compute_grade(
        price=102.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5,
    )
    assert result.distance_atr == pytest.approx(1.0)


def test_grade_result_distance_atr_is_none_without_a_valid_atr14():
    result = le.compute_grade(
        price=100.0, atr14=None, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5,
    )
    assert result.distance_atr is None


def test_apply_portfolio_grade_modifiers_preserves_distance_atr():
    base = le.GradeResult(grade=le.Grade.A, reasons=[], distance_atr=0.42)
    result = le.apply_portfolio_grade_modifiers(base, max_correlation_with_open_position=0.9)
    assert result.distance_atr == pytest.approx(0.42)


def test_grade_distance_beyond_a_threshold_falls_back_to_b():
    # Mismo perfil de grado A salvo que el precio está a 1.2 ATR del nivel
    # (por encima del techo de A, 1.0, pero dentro del de B, 1.5).
    result = le.compute_grade(
        price=102.4, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5,
    )
    assert result.grade == le.Grade.B


def test_grade_high_relative_strength_upgrades_one_step():
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=0.8, rs_percentile=80,
    )
    assert result.grade == le.Grade.A  # B (por el volumen bajo) sube a A
    assert any("Fuerza relativa alta" in r for r in result.reasons)


def test_grade_upgrade_has_no_effect_when_already_at_the_ceiling():
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5, rs_percentile=90,
    )
    assert result.grade == le.Grade.A


def test_grade_low_relative_strength_caps_at_b():
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5, rs_percentile=30,
    )
    assert result.grade == le.Grade.B
    assert any("Fuerza relativa baja" in r for r in result.reasons)


def test_grade_price_below_sma200_caps_at_b():
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5, sma200=110.0,
    )
    assert result.grade == le.Grade.B
    assert any("SMA200" in r for r in result.reasons)


def test_grade_strong_sector_upgrades_one_step():
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=0.8, sector_rs_percentile=75,
    )
    assert result.grade == le.Grade.A
    assert any("Sector fuerte" in r for r in result.reasons)


def test_grade_weak_sector_caps_at_b():
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=1.5, sector_rs_percentile=20,
    )
    assert result.grade == le.Grade.B
    assert any("Sector débil" in r for r in result.reasons)


def test_grade_never_upgrades_more_than_one_step_from_multiple_reasons():
    # Grado base B (volumen bajo) + fuerza relativa alta Y sector fuerte a la
    # vez - solo un escalón, no dos: sigue en A, no hay grado por encima.
    result = le.compute_grade(
        price=100.0, atr14=2.0, entry_trigger=_trigger(100.0), geometry=_geometry(risk_atr=1.0),
        weekly_bullish=True, relative_volume=0.8, rs_percentile=90, sector_rs_percentile=90,
    )
    assert result.grade == le.Grade.A
    upgrade_reasons = sum(("Fuerza relativa alta" in r or "Sector fuerte" in r) for r in result.reasons)
    assert upgrade_reasons == 2


def test_apply_portfolio_grade_modifiers_high_correlation_forces_c():
    base = le.GradeResult(grade=le.Grade.A, reasons=[])
    result = le.apply_portfolio_grade_modifiers(base, max_correlation_with_open_position=0.85)
    assert result.grade == le.Grade.C
    assert any("Correlación alta" in r for r in result.reasons)


def test_apply_portfolio_grade_modifiers_low_correlation_is_a_noop():
    base = le.GradeResult(grade=le.Grade.A, reasons=[])
    result = le.apply_portfolio_grade_modifiers(base, max_correlation_with_open_position=0.3)
    assert result.grade == le.Grade.A
    assert result.reasons == []


def test_apply_portfolio_grade_modifiers_full_portfolio_annotates_without_downgrading():
    base = le.GradeResult(grade=le.Grade.A, reasons=[])
    result = le.apply_portfolio_grade_modifiers(base, open_positions_count=10, max_open_positions=10)
    assert result.grade == le.Grade.A  # no baja el grado, literal
    assert any("Cartera llena" in r for r in result.reasons)


def test_apply_portfolio_grade_modifiers_is_a_noop_on_no_grade():
    base = le.GradeResult(grade=None, reasons=["geometría no viable"])
    result = le.apply_portfolio_grade_modifiers(base, max_correlation_with_open_position=0.9)
    assert result.grade is None
    assert result.reasons == ["geometría no viable"]


def test_gate_grade_and_geometry_functions_never_accept_a_monthly_timeframe_reading():
    """Parte 7.2 (biblioteca de setups, quant_methodology.md §28.x), literal:
    "la celda mensual no entra en el gate, ni en el grado, ni en ningún
    gatillo". El encargo pide un test que compare dos escenarios idénticos
    salvo por la lectura mensual - pero eso resultó impracticable de
    construir de verdad: `classify_stage` reutiliza el mismo `lookback=20`/
    `long_lookback=100` (en unidades de la propia serie) para mensual Y
    semanal, así que cualquier tramo de historial lo bastante antiguo para
    mover la etapa mensual también cae dentro del alcance de
    `long_lookback` de la etapa SEMANAL (verificado con un script) - no hay
    forma de que solo uno de los dos difiera con una construcción simple de
    "tramo reciente idéntico, tramo antiguo distinto". La garantía real y
    verificable es estructural, no empírica: ninguna de estas cuatro
    funciones - las únicas que deciden gate/grado/geometría/tamaño en todo
    el sistema - tiene siquiera un parámetro por el que
    `multi_timeframe.TimeframeStrip`/su celda mensual podría entrar. Mismo
    principio que `test_exit_engine_never_imports_recommendation_engine`,
    aplicado a la firma en vez de a los imports."""
    for func in (le.evaluate_gate, le.compute_grade, tg.compute_entry_geometry, tg.size_position):
        for name, param in inspect.signature(func).parameters.items():
            assert "monthly" not in name.lower()
            assert "timeframe_strip" not in name.lower()
            assert "TimeframeStrip" not in str(param.annotation)
