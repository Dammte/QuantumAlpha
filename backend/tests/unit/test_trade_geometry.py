import pytest

from app.services import trade_geometry as tg
from app.services.technical_analysis import Level, LevelKind, LevelState, PriceLevel, TrendState

# --- compute_entry_trigger ---------------------------------------------------


def test_pullback_trigger_when_support_within_proximity():
    support = PriceLevel(price=97.0, kind="support", strength=2, distance_pct=-0.02)  # 2% below, within 3%
    trigger = tg.compute_entry_trigger(price=99.0, nearest_support=support, nearest_resistance=None)
    assert trigger.trigger_type == "pullback_bounce"
    assert trigger.trigger_price == pytest.approx(97.0)
    assert trigger.already_triggered is True


def test_pullback_trigger_exactly_at_proximity_boundary():
    # abs(distance_pct) == PULLBACK_PROXIMITY_PCT exactly - boundary is inclusive.
    support = PriceLevel(price=97.0, kind="support", strength=1, distance_pct=-tg.PULLBACK_PROXIMITY_PCT)
    trigger = tg.compute_entry_trigger(price=100.0, nearest_support=support, nearest_resistance=None)
    assert trigger is not None
    assert trigger.trigger_type == "pullback_bounce"


def test_no_pullback_trigger_when_support_too_far():
    support = PriceLevel(price=95.0, kind="support", strength=2, distance_pct=-0.05)  # 5% below, beyond 3%
    trigger = tg.compute_entry_trigger(price=100.0, nearest_support=support, nearest_resistance=None)
    assert trigger is None


def test_breakout_trigger_when_resistance_within_max_distance():
    resistance = PriceLevel(price=105.0, kind="resistance", strength=2, distance_pct=0.05)  # 5%, within 8%
    trigger = tg.compute_entry_trigger(price=100.0, nearest_support=None, nearest_resistance=resistance)
    assert trigger.trigger_type == "breakout"
    assert trigger.trigger_price == pytest.approx(105.0 * 1.003)
    assert trigger.already_triggered is False  # price (100) hasn't reached the trigger (~105.3) yet


def test_breakout_trigger_exactly_at_max_distance_boundary():
    resistance = PriceLevel(
        price=108.0, kind="resistance", strength=1, distance_pct=tg.BREAKOUT_TRIGGER_MAX_DISTANCE
    )
    trigger = tg.compute_entry_trigger(price=100.0, nearest_support=None, nearest_resistance=resistance)
    assert trigger is not None
    assert trigger.trigger_type == "breakout"


def test_no_breakout_trigger_when_resistance_too_far():
    resistance = PriceLevel(price=115.0, kind="resistance", strength=2, distance_pct=0.15)  # 15%, beyond 8%
    trigger = tg.compute_entry_trigger(price=100.0, nearest_support=None, nearest_resistance=resistance)
    assert trigger is None


def test_no_trigger_when_resistance_already_behind_price():
    # distance_pct <= 0 means this "resistance" pivot is no longer above price -
    # already broken, not an upcoming trigger.
    resistance = PriceLevel(price=99.0, kind="resistance", strength=1, distance_pct=-0.01)
    trigger = tg.compute_entry_trigger(price=100.0, nearest_support=None, nearest_resistance=resistance)
    assert trigger is None


def test_no_trigger_when_neither_level_present():
    assert tg.compute_entry_trigger(price=100.0, nearest_support=None, nearest_resistance=None) is None


def test_pullback_takes_priority_over_breakout_when_both_present():
    support = PriceLevel(price=98.0, kind="support", strength=2, distance_pct=-0.02)
    resistance = PriceLevel(price=104.0, kind="resistance", strength=2, distance_pct=0.04)
    trigger = tg.compute_entry_trigger(price=100.0, nearest_support=support, nearest_resistance=resistance)
    assert trigger.trigger_type == "pullback_bounce"


def test_breakout_already_triggered_when_a_fresh_price_has_cleared_a_stale_level():
    # Models the intraday-refresh use case: nearest_resistance was computed
    # against yesterday's close (100 -> distance_pct 0.05, resistance at 105),
    # but `price` here is a fresh intraday quote that has already cleared the
    # breakout buffer (105 * 1.003 = 105.315) - already_triggered must reflect
    # today's live price, not the stale snapshot's own price.
    resistance = PriceLevel(price=105.0, kind="resistance", strength=2, distance_pct=0.05)
    trigger = tg.compute_entry_trigger(price=106.0, nearest_support=None, nearest_resistance=resistance)
    assert trigger.already_triggered is True


# --- compute_stop_and_target (moved unchanged from recommendation_engine.py) -


def test_compute_stop_and_target_none_without_atr():
    result = tg.compute_stop_and_target(price=100.0, atr14=None, nearest_support=None, nearest_resistance=None)
    assert result == tg.StopAndTarget(None, None, None, None)


def test_compute_stop_and_target_uses_atr_ceiling_without_a_nearby_support():
    result = tg.compute_stop_and_target(price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None)
    assert result.stop_loss == pytest.approx(100.0 - tg.STOP_ATR_CEILING * 2.0)
    assert result.take_profit is not None
    assert result.take_profit_method == f"objetivo {tg.REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"


def test_compute_stop_and_target_prefers_the_tighter_of_support_and_atr_ceiling():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_stop_and_target(price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None)
    assert result.stop_loss == pytest.approx(99.0 * 0.99)


def test_compute_stop_and_target_targets_resistance_when_reward_risk_clears_the_bar():
    support = PriceLevel(price=97.0, kind="support", strength=2, distance_pct=-0.03)
    resistance = PriceLevel(price=110.0, kind="resistance", strength=1, distance_pct=0.10)
    result = tg.compute_stop_and_target(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=resistance
    )
    assert result.take_profit == pytest.approx(110.0)
    assert result.take_profit_method == "resistencia más cercana"


def test_compute_stop_and_target_none_take_profit_when_stop_is_at_or_above_price():
    support = PriceLevel(price=102.0, kind="support", strength=1, distance_pct=0.02)
    result = tg.compute_stop_and_target(price=100.0, atr14=0.001, nearest_support=support, nearest_resistance=None)
    assert result.stop_loss is not None
    assert result.take_profit is None
    assert result.risk_reward is None


# --- compute_entry_geometry / size_position: the split behind ---------------
# --- compute_trade_geometry (see the module's own docstring for why) --------


def test_entry_geometry_never_computes_sizing_fields():
    # No capital_total parameter at all - evaluate_gate/daily_close.py score
    # the whole universe with no portfolio in scope, so these three fields
    # must always come back None here regardless of how viable the setup is.
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    assert result.viable is True
    assert result.shares_for_risk_budget is None
    assert result.position_value is None
    assert result.pct_of_portfolio is None


def test_size_position_fills_in_the_sizing_fields_of_a_viable_geometry():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    geometry = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    sized = tg.size_position(geometry, capital_total=100_000.0)
    assert sized.shares_for_risk_budget == pytest.approx(150.0)
    assert sized.position_value == pytest.approx(15_000.0)
    assert sized.pct_of_portfolio == pytest.approx(0.15)
    # Everything from the entry geometry itself is carried through unchanged.
    assert sized.stop_price == geometry.stop_price
    assert sized.entry_type == geometry.entry_type


def test_size_position_is_a_no_op_on_a_rejected_geometry():
    # No cascade rung applies - viable=False, nothing to size.
    geometry = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    assert geometry.viable is False
    sized = tg.size_position(geometry, capital_total=100_000.0)
    assert sized == geometry


def test_size_position_can_still_reject_on_minimum_position_value():
    # A viable entry geometry, but too little capital to size a real position.
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    geometry = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    assert geometry.viable is True
    sized = tg.size_position(geometry, capital_total=500.0)
    assert sized.viable is False
    assert sized.position_value == pytest.approx(75.0)
    assert "pequeña" in sized.rejection_reason


def test_compute_trade_geometry_matches_the_two_step_split():
    # The convenience wrapper must be exactly equivalent to calling the two
    # halves separately - no divergent behavior hiding in the wrapper itself.
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    wrapped = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
        atr_percentile_252=0.9,
    )
    geometry = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    two_step = tg.size_position(geometry, capital_total=100_000.0, atr_percentile_252=0.9)
    assert wrapped == two_step


def test_geometry_to_dict_and_back_round_trips_a_viable_geometry():
    # `daily_close.py` persists exactly this dict shape on `TickerDailyState`
    # (Parte 7, later pass) - a caller with a specific portfolio's capital
    # (e.g. `GET /market/radar?portfolio_id=`) must get back a real
    # `TradeGeometry` it can run through `size_position`, not just a display
    # shape.
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    geometry = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    assert geometry.viable is True

    data = tg.geometry_to_dict(geometry)
    assert data["entry_type"] == "pullback_support"
    restored = tg.geometry_from_dict(data)
    assert restored == geometry

    sized = tg.size_position(restored, capital_total=100_000.0)
    assert sized.shares_for_risk_budget is not None


def test_geometry_to_dict_serializes_a_none_entry_type():
    geometry = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    assert geometry.viable is False
    data = tg.geometry_to_dict(geometry)
    assert data["entry_type"] is None
    assert tg.geometry_from_dict(data) == geometry


# --- compute_trade_geometry: the real Parte 7 design (stop cascade + ---------
# --- adaptive risk ceiling + cost-net target + fixed-risk sizing) -----------
#
# capital_total=100_000.0 throughout unless a test is specifically about
# sizing, so RISK_PER_TRADE_PCT (1%) and MAX_POSITION_PCT (15%) produce clean
# numbers; every scenario keeps net reward:risk comfortably above
# MIN_RISK_REWARD_NET (1.5) unless that's the exact thing under test.


def test_geometry_none_when_atr_is_unavailable():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=None, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.viable is False
    assert "ATR" in result.rejection_reason


def test_geometry_not_viable_when_no_cascade_rung_applies():
    # Sideways, no nearby support/resistance, no EMA reads - nothing to anchor a stop to.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.viable is False
    assert result.entry_type is None
    assert "nivel de referencia" in result.rejection_reason


def test_geometry_ema_rungs_never_apply_outside_an_uptrend():
    # Same EMA21/55 reads as the pullback/continuation tests below, but in a
    # downtrend - Parte 7's rungs 3/4 are explicitly "en tendencia"/"continuación",
    # never a downtrend reference.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=99.0, ema55=95.0, trend=TrendState.DOWNTREND, capital_total=100_000.0,
    )
    assert result.viable is False
    assert result.entry_type is None


# --- stop cascade, one rung per entry type -----------------------------------
# Todas las geometrías de esta sección tienen atr_pct = atr14/price = 2% con
# price=100.0/atr14=2.0, salvo que se diga lo contrario - justo en el perfil
# "normal" (< VOLATILITY_PROFILE_NORMAL_MAX_ATR_PCT=4%), colchón
# STOP_CUSHION_ATR_NORMAL=0.35 (ver `classify_volatility_profile`).


def _broken_resistance(price: float) -> Level:
    """Auditoria del Radar, bloque H1/H2: el único anclaje de ruptura válido
    ahora es un `Level` con estado `BROKEN_CONFIRMED` - una `PriceLevel` de
    resistencia (por construcción, siempre por ENCIMA del precio) ya no basta
    ni debe bastar para simular una ruptura ya confirmada."""
    return Level(
        kind=LevelKind.PIVOT_RESISTANCE, price=price, side="above", distance_pct=-0.02,
        distance_atr=-1.0, state=LevelState.BROKEN_CONFIRMED, bars_in_state=1, strength=2, slope_pct_20d=None,
    )


def test_geometry_breakout_rung_stop_below_the_broken_resistance():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
        levels=[_broken_resistance(98.0)],
    )
    assert result.entry_type == tg.EntryType.BREAKOUT
    assert result.level_kind == LevelKind.PIVOT_RESISTANCE
    assert result.stop_price == pytest.approx(97.3)  # 98.0 - 0.35*2.0 (perfil normal)
    assert result.risk_atr == pytest.approx(1.35)
    assert "resistencia roto" in result.stop_basis
    assert result.viable is True


def test_geometry_no_breakout_rung_without_a_confirmed_broken_level():
    # Bloque H1: una `PriceLevel` de resistencia todavía por encima del precio
    # NUNCA basta para el peldaño de ruptura (el bug estructural que hacía
    # que ese peldaño fuera inalcanzable) - sin `levels`, ni siquiera pasa el
    # "casi rota" de antes.
    resistance = PriceLevel(price=105.0, kind="resistance", strength=2, distance_pct=0.05)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.entry_type is None
    assert result.viable is False


def test_geometry_bounce_rung_stop_below_the_pivot_low():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.entry_type == tg.EntryType.PULLBACK_SUPPORT
    assert result.level_kind == LevelKind.PIVOT_SUPPORT
    assert result.stop_price == pytest.approx(98.3)  # 99.0 - 0.35*2.0
    assert "soporte" in result.stop_basis
    assert result.viable is True


def test_geometry_bounce_rung_requires_support_within_pullback_proximity():
    # 6% away - too far to count as "at" the support (PULLBACK_PROXIMITY_PCT is 3%).
    support = PriceLevel(price=94.0, kind="support", strength=2, distance_pct=-0.06)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.entry_type is None
    assert result.viable is False


def test_geometry_breakout_rung_takes_priority_over_bounce_rung():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
        levels=[_broken_resistance(98.0)],
    )
    assert result.entry_type == tg.EntryType.BREAKOUT


def test_geometry_pullback_ema21_rung_in_an_uptrend():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=99.0, ema55=90.0, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.entry_type == tg.EntryType.PULLBACK_EMA21
    assert result.level_kind == LevelKind.EMA21
    assert result.stop_price == pytest.approx(98.3)  # 99.0 - 0.35*2.0 (perfil normal)
    assert "EMA21" in result.stop_basis
    assert result.viable is True


def test_geometry_continuation_ema55_rung_when_ema21_is_too_far():
    # Price is well clear of EMA21 (5% away, outside PULLBACK_PROXIMITY_PCT),
    # so rung 3 doesn't apply - falls through to the EMA55 continuation rung.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=95.0, ema55=99.0, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.entry_type == tg.EntryType.CONTINUATION_EMA55
    assert result.level_kind == LevelKind.EMA55
    assert result.stop_price == pytest.approx(98.3)  # 99.0 - 0.35*2.0
    assert "EMA55" in result.stop_basis
    assert result.viable is True


def test_geometry_range_low_20_is_the_last_resort_anchor():
    # Bloque H2, paso 2, literal: "sin estructura cercana válida -> bajo el
    # mínimo de 20 sesiones" - el único peldaño que no depende del tipo de
    # entrada, ofrecido cuando nada más (soporte/resistencia/EMA) aplica.
    range_low = Level(
        kind=LevelKind.RANGE_LOW_20, price=95.0, side="above", distance_pct=-0.05,
        distance_atr=2.5, state=LevelState.FAR, bars_in_state=5, strength=None, slope_pct_20d=None,
    )
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
        levels=[range_low],
    )
    assert result.entry_type == tg.EntryType.PULLBACK_SUPPORT
    assert result.level_kind == LevelKind.RANGE_LOW_20
    assert result.stop_price == pytest.approx(94.3)  # 95.0 - 0.35*2.0
    assert "mínimo de 20 sesiones" in result.stop_basis
    assert result.viable is True


# --- "demasiado cerca" escala al siguiente peldaño (bloque H2, paso 4.3) -----


def test_geometry_skips_a_too_close_anchor_and_escalates_to_the_next_rung():
    # El soporte está a solo 0.3 puntos de distancia natural - con el colchón
    # normal (0.35*2.0=0.7) el stop resultante queda a 0.5 ATR, por debajo del
    # mínimo STOP_MIN_DISTANCE_ATR=0.8 - ruido, no nivel. La cascada debe
    # descartarlo y probar el siguiente peldaño (EMA55) en vez de aceptar un
    # stop pegado al precio o rechazar la operación entera.
    support = PriceLevel(price=99.7, kind="support", strength=2, distance_pct=-0.003)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=95.0, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.entry_type == tg.EntryType.CONTINUATION_EMA55
    assert result.level_kind == LevelKind.EMA55
    assert result.stop_price == pytest.approx(94.3)  # 95.0 - 0.35*2.0, no el soporte descartado
    assert result.risk_atr == pytest.approx(2.85)
    assert result.viable is True


def test_geometry_rejects_when_every_candidate_anchor_is_too_close():
    support = PriceLevel(price=99.7, kind="support", strength=2, distance_pct=-0.003)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.viable is False
    assert "demasiado cerca" in result.rejection_reason


# --- el stop ya no se mueve para caber (bloque H2, paso 4: "el stop no se ---
# --- mueve para caber; el tamaño sí") - ni el techo duro de 2.0 ATR ni el ---
# --- techo de riesgo adaptativo rechazan o encogen el stop; solo informan --


def test_geometry_no_longer_caps_the_stop_at_the_hard_atr_ceiling():
    # Un nombre muy tranquilo (ATR 0.5% del precio, perfil "tranquilo",
    # colchón 0.25) cuyo stop natural pide casi 3.0 ATR de riesgo - antes se
    # habría recortado a 2.0 ATR exactos; ahora se deja tal cual, más ancho
    # que el viejo techo duro, y es `size_position` quien absorbería ese
    # riesgo reduciendo acciones, no esta función.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=0.5, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
        levels=[_broken_resistance(98.65)],
    )
    assert result.viable is True
    assert result.risk_atr == pytest.approx(2.95)  # por encima del viejo techo duro de 2.0
    assert result.stop_price == pytest.approx(98.525)  # 98.65 - 0.25*0.5, sin recortar
    assert "techo" not in result.stop_basis


# --- techo de riesgo adaptativo, ahora puramente informativo (bloque H2) -----


def test_geometry_informational_risk_ceiling_no_longer_rejects_a_wide_stop():
    # Perfil "tranquilo" (ATR 1.2% del precio) -> techo informativo
    # clamp(2.5*1.2%, 2%, 7%) = 3.0%. El stop natural (peldaño EMA55) pide un
    # riesgo mayor a ese techo - bloque H2: ya no se rechaza por eso, el
    # techo solo se expone para que `size_position`/la UI lo muestren.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=1.2, nearest_support=None, nearest_resistance=None,
        ema21=90.0, ema55=96.48, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.viable is True
    assert result.risk_ceiling_pct == pytest.approx(0.03)
    assert result.risk_pct == pytest.approx(0.0382, rel=1e-3)  # 96.48 - 0.25*1.2 = 96.18


def test_geometry_volatile_profile_ceiling_clamped_at_its_max():
    # ATR 5.0% del precio -> perfil "volátil", techo informativo
    # clamp(2.5*5%, 2%, 7%) = 7.0% (tope del clamp).
    result = tg.compute_trade_geometry(
        price=100.0, atr14=5.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=95.4, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.viable is True
    assert result.risk_ceiling_pct == pytest.approx(0.07)
    assert result.risk_pct == pytest.approx(0.071)  # 95.4 - 0.5*5.0 = 92.9 -> riesgo 7.1%


def test_geometry_normal_large_cap_profile_ceiling():
    # ATR 2.0% of price -> ceiling clamp(2.5*2%, 2%, 7%) = 5.0%.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
        levels=[_broken_resistance(98.0)],
    )
    assert result.risk_ceiling_pct == pytest.approx(0.05)


def test_geometry_erratic_microcap_profile_ceiling_no_longer_rejects():
    # ATR 6.0% of price -> perfil "volátil", clamp(2.5*6%, 2%, 7%) = 7.0% (el
    # tope del clamp, no el 15% sin acotar) - antes esto rechazaba la
    # operación; ahora solo se informa, y la operación es viable.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=6.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=92.0, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.risk_ceiling_pct == pytest.approx(0.07)
    # Stop natural: 92.0 - 0.5*6.0 = 89.0 -> 11% de riesgo, por encima del techo informativo.
    assert result.viable is True
    assert result.risk_pct == pytest.approx(0.11)


# --- target selection: resistance vs fixed 2R vs rejection -------------------


def test_geometry_targets_the_nearest_resistance_when_its_net_rr_clears_the_bar():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    resistance = PriceLevel(price=103.0, kind="resistance", strength=2, distance_pct=0.03)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.target_price == pytest.approx(103.0)
    assert result.target_basis == "resistencia más cercana"
    assert result.risk_reward_net is not None and result.risk_reward_net >= tg.MIN_RISK_REWARD_NET
    assert result.risk_reward_net < result.risk_reward_gross  # net is always tighter than gross


def test_geometry_falls_back_to_fixed_target_when_resistance_net_rr_is_too_low():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    resistance = PriceLevel(price=101.5, kind="resistance", strength=2, distance_pct=0.015)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.target_basis == f"objetivo {tg.REWARD_RISK_RATIO:.0f}:1 sobre el riesgo"
    assert result.risk_reward_gross == pytest.approx(2.0)
    assert result.viable is True


def test_geometry_rejected_when_even_the_fixed_target_cannot_clear_costs():
    # Un riesgo por acción pequeño (pero todavía por encima de
    # STOP_MIN_DISTANCE_ATR, para que el rechazo venga del objetivo y no del
    # guardarraíl de "demasiado cerca") - el objetivo fijo 2:1 tiene un R/R
    # *bruto* de exactamente 2.0, pero el coste de ida y vuelta es grande
    # relativo a una recompensa tan pequeña, así que el R/R *neto* cae por
    # debajo de MIN_RISK_REWARD_NET.
    support = PriceLevel(price=99.8, kind="support", strength=2, distance_pct=-0.002)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=0.3, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.risk_atr == pytest.approx(0.9167, rel=1e-3)  # por encima de STOP_MIN_DISTANCE_ATR=0.8
    assert result.viable is False
    assert result.target_price is None
    assert "costes" in result.rejection_reason


# --- sizing: the three limits -------------------------------------------------


def test_geometry_sizing_uses_risk_per_trade_pct_of_capital():
    # atr_pct=3% -> perfil "normal", colchón 0.35: stop = 95.0-0.35*3.0=93.95,
    # riesgo=6.05% - ancho de sobra para que RISK_PER_TRADE_PCT's own sizing
    # (1% / 6.05% = ~16.5% of capital) todavía dispare MAX_POSITION_PCT (15%),
    # con margen para que la prueba de halving de abajo muestre una
    # diferencia real, sin recorte.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=3.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=95.0, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.risk_pct == pytest.approx(0.0605)
    # Risk-based size alone sería (100_000*0.01)/6.05 = 165.29 acciones - MAX_POSITION_PCT lo recorta.
    assert result.shares_for_risk_budget == pytest.approx(150.0)
    assert result.position_value == pytest.approx(15_000.0)
    assert result.pct_of_portfolio == pytest.approx(0.15)


def test_geometry_sizing_halves_at_or_above_the_85th_atr_percentile():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=3.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=95.0, trend=TrendState.UPTREND, capital_total=100_000.0,
        atr_percentile_252=0.9,
    )
    # Halved *before* the MAX_POSITION_PCT cap is applied - (100_000*0.01)/6.05/2 = 82.64,
    # which no longer needs capping at all (a real, visible halving, not masked by the cap).
    assert result.shares_for_risk_budget == pytest.approx(82.645, rel=1e-3)
    assert result.position_value == pytest.approx(8_264.46, rel=1e-3)


def test_geometry_sizing_does_not_halve_below_the_85th_atr_percentile():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=3.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=95.0, trend=TrendState.UPTREND, capital_total=100_000.0,
        atr_percentile_252=0.84,
    )
    assert result.shares_for_risk_budget == pytest.approx(150.0)


def test_geometry_sizing_capped_at_max_position_pct():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=1_000_000.0,
    )
    assert result.pct_of_portfolio == pytest.approx(tg.MAX_POSITION_PCT)
    assert result.position_value == pytest.approx(1_000_000.0 * tg.MAX_POSITION_PCT)


def test_geometry_rejected_when_position_value_is_below_the_minimum():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=500.0,
    )
    assert result.viable is False
    assert result.position_value == pytest.approx(75.0)
    assert "pequeña" in result.rejection_reason


def test_geometry_accepted_just_above_the_minimum_position_value():
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=534.0,
    )
    assert result.viable is True
    assert result.position_value == pytest.approx(80.1)
