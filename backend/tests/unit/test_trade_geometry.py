import pytest

from app.services import trade_geometry as tg
from app.services.technical_analysis import PriceLevel, TrendState

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
    assert result.stop_loss == pytest.approx(100.0 - tg.ATR_STOP_MULTIPLE * 2.0)
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
    resistance = PriceLevel(price=98.0, kind="resistance", strength=2, distance_pct=-0.02)
    result = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS,
    )
    assert result.viable is True
    assert result.shares_for_risk_budget is None
    assert result.position_value is None
    assert result.pct_of_portfolio is None


def test_size_position_fills_in_the_sizing_fields_of_a_viable_geometry():
    resistance = PriceLevel(price=98.0, kind="resistance", strength=2, distance_pct=-0.02)
    geometry = tg.compute_entry_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=resistance,
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


def test_geometry_breakout_rung_stop_below_the_broken_resistance():
    resistance = PriceLevel(price=98.0, kind="resistance", strength=2, distance_pct=-0.02)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.entry_type == tg.EntryType.BREAKOUT
    # 98.0 - 0.3*2.0 (LEVEL_STOP_CUSHION_ATR)
    assert result.stop_price == pytest.approx(97.4)
    assert result.risk_atr == pytest.approx(1.3)
    assert "resistencia roto" in result.stop_basis
    assert result.viable is True


def test_geometry_no_breakout_rung_before_the_resistance_is_actually_cleared():
    # Price hasn't reached the breakout buffer yet - falls through to "no rung applies".
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
    assert result.stop_price == pytest.approx(98.4)  # 99.0 - 0.3*2.0
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
    resistance = PriceLevel(price=98.0, kind="resistance", strength=2, distance_pct=-0.02)
    support = PriceLevel(price=99.0, kind="support", strength=2, distance_pct=-0.01)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=support, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.entry_type == tg.EntryType.BREAKOUT


def test_geometry_pullback_ema21_rung_in_an_uptrend():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=None,
        ema21=99.0, ema55=90.0, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.entry_type == tg.EntryType.PULLBACK_EMA21
    assert result.stop_price == pytest.approx(98.2)  # 99.0 - 0.4*2.0 (MA_STOP_CUSHION_ATR)
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
    assert result.stop_price == pytest.approx(98.2)  # 99.0 - 0.4*2.0
    assert "EMA55" in result.stop_basis
    assert result.viable is True


# --- hard 2.0 ATR stop ceiling -----------------------------------------------


def test_geometry_hard_atr_ceiling_caps_a_stop_that_demands_more():
    # A very calm name (ATR = 0.5% of price) whose adaptive risk ceiling is
    # floor-clamped to 2% - loose enough that the *hard* 2.0 ATR cap is what
    # actually binds here, not the risk ceiling (see the module's own
    # docstring on why the risk ceiling is checked against the *natural*,
    # uncapped distance first).
    resistance = PriceLevel(price=98.65, kind="resistance", strength=2, distance_pct=-0.0135)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=0.5, nearest_support=None, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.viable is True
    assert result.risk_atr == pytest.approx(2.0)
    assert result.stop_price == pytest.approx(99.0)  # 100 - 2.0*0.5, not the natural 98.56
    assert "techo de 2.0 ATR" in result.stop_basis


# --- adaptive risk ceiling, per volatility profile (Parte 15/20) -------------


def test_geometry_calm_utility_profile_rejects_a_natural_stop_beyond_its_ceiling():
    # Parte 15 escenario #6: ATR 1.2% of price -> ceiling clamp(2.5*1.2%, 2%,
    # 7%) = 3.0%. A natural (EMA55-continuation) stop asking for 4% is
    # rejected - even though it's well under the *hard* 2.0 ATR cap
    # (4%/1.2% = 3.33 ATR... wait, that one *would* exceed 2.0 ATR too, but
    # the risk ceiling (checked first, against the natural distance) already
    # rejects it before the hard cap is ever applied).
    result = tg.compute_trade_geometry(
        price=100.0, atr14=1.2, nearest_support=None, nearest_resistance=None,
        ema21=90.0, ema55=96.48, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.viable is False
    assert result.risk_ceiling_pct == pytest.approx(0.03)
    assert result.risk_pct == pytest.approx(0.04)
    assert "techo adaptativo" in result.rejection_reason


def test_geometry_volatile_semiconductor_profile_accepts_within_its_wider_ceiling():
    # Parte 15 escenario #7: ATR 3.5% of price -> ceiling clamp(2.5*3.5%, 2%,
    # 7%) = 7.0% (hits the max clamp). A 6% natural stop is accepted.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=3.5, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=95.4, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.viable is True
    assert result.risk_ceiling_pct == pytest.approx(0.07)
    assert result.risk_pct == pytest.approx(0.06)


def test_geometry_normal_large_cap_profile_ceiling():
    # ATR 2.0% of price -> ceiling clamp(2.5*2%, 2%, 7%) = 5.0%.
    resistance = PriceLevel(price=98.0, kind="resistance", strength=2, distance_pct=-0.02)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=2.0, nearest_support=None, nearest_resistance=resistance,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.risk_ceiling_pct == pytest.approx(0.05)


def test_geometry_erratic_microcap_profile_ceiling_is_capped_not_15_percent():
    # ATR 6.0% of price -> clamp(2.5*6%, 2%, 7%) = 7.0% (the max clamp, not
    # the unclamped 15%) - Parte 20's own "often will reject" case: a natural
    # stop only slightly beyond 7% (well within what 6%-ATR noise produces)
    # already exceeds it.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=6.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=92.0, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.risk_ceiling_pct == pytest.approx(0.07)
    # Natural stop: 92.0 - 0.4*6.0 = 89.6 -> 10.4% risk, well over the 7% ceiling.
    assert result.viable is False
    assert result.risk_pct == pytest.approx(0.104)


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
    # A tiny risk-per-share (a very tight stop on a very calm name) - the
    # fixed 2:1 target's *gross* R/R is still exactly 2.0, but the flat
    # round-trip transaction cost is large relative to such a small reward,
    # so the *net* R/R falls under MIN_RISK_REWARD_NET.
    support = PriceLevel(price=99.95, kind="support", strength=2, distance_pct=-0.0005)
    result = tg.compute_trade_geometry(
        price=100.0, atr14=0.3, nearest_support=support, nearest_resistance=None,
        ema21=None, ema55=None, trend=TrendState.SIDEWAYS, capital_total=100_000.0,
    )
    assert result.viable is False
    assert result.target_price is None
    assert "costes" in result.rejection_reason or "cerca" in result.rejection_reason


# --- sizing: the three limits -------------------------------------------------


def test_geometry_sizing_uses_risk_per_trade_pct_of_capital():
    # Wide enough risk (6%) that RISK_PER_TRADE_PCT's own sizing (1% / 6% =
    # ~16.7% of capital) still clears MAX_POSITION_PCT (15%) with room for
    # the halving test below to show a real, uncapped difference.
    result = tg.compute_trade_geometry(
        price=100.0, atr14=3.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=94.2, trend=TrendState.UPTREND, capital_total=100_000.0,
    )
    assert result.risk_pct == pytest.approx(0.06)
    # Risk-based size alone would be (100_000*0.01)/6.0 = 166.67 shares - MAX_POSITION_PCT caps it.
    assert result.shares_for_risk_budget == pytest.approx(150.0)
    assert result.position_value == pytest.approx(15_000.0)
    assert result.pct_of_portfolio == pytest.approx(0.15)


def test_geometry_sizing_halves_at_or_above_the_85th_atr_percentile():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=3.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=94.2, trend=TrendState.UPTREND, capital_total=100_000.0,
        atr_percentile_252=0.9,
    )
    # Halved *before* the MAX_POSITION_PCT cap is applied - (100_000*0.01)/6.0/2 = 83.33,
    # which no longer needs capping at all (a real, visible halving, not masked by the cap).
    assert result.shares_for_risk_budget == pytest.approx(83.333, rel=1e-3)
    assert result.position_value == pytest.approx(8_333.33, rel=1e-3)


def test_geometry_sizing_does_not_halve_below_the_85th_atr_percentile():
    result = tg.compute_trade_geometry(
        price=100.0, atr14=3.0, nearest_support=None, nearest_resistance=None,
        ema21=80.0, ema55=94.2, trend=TrendState.UPTREND, capital_total=100_000.0,
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
