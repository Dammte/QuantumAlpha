import pandas as pd
import pytest

from app.services import trade_manager as tm

# --- chandelier_stop ---------------------------------------------------------


def test_chandelier_stop_basic():
    high = pd.Series([100.0, 102.0, 105.0, 103.0, 101.0])
    atr14 = pd.Series([2.0] * 5)
    result = tm.chandelier_stop(high, atr14, multiplier=3.0, window=5)
    assert result == pytest.approx(105.0 - 3 * 2.0)


def test_chandelier_stop_uses_all_available_bars_when_younger_than_the_window():
    # A position younger than `window` bars (e.g. just opened, or entry-
    # bounded `high` from a young trade_plan) still gets a real read from
    # whatever history it has - it no longer demands a full window-bar
    # history before answering (see the function's own docstring for why: a
    # fixed-window read applied to history reaching before the position's
    # own entry is exactly what let a young position's stop sit above a
    # pre-entry high).
    high = pd.Series([100.0, 102.0])
    atr14 = pd.Series([2.0] * 2)
    result = tm.chandelier_stop(high, atr14, multiplier=3.0, window=5)
    assert result == pytest.approx(102.0 - 3 * 2.0)


def test_chandelier_stop_none_when_high_is_empty():
    high = pd.Series(dtype=float)
    atr14 = pd.Series([2.0] * 5)
    assert tm.chandelier_stop(high, atr14, multiplier=3.0, window=5) is None


def test_chandelier_stop_none_with_empty_atr():
    high = pd.Series([100.0] * 10)
    atr14 = pd.Series(dtype=float)
    assert tm.chandelier_stop(high, atr14, multiplier=3.0, window=5) is None


# --- chandelier_multiplier ----------------------------------------------------


def test_chandelier_multiplier_by_regime():
    assert tm.chandelier_multiplier("baja", r_multiple=0.5) == tm.CHANDELIER_MULTIPLIER_BY_REGIME["baja"]
    assert tm.chandelier_multiplier("alta", r_multiple=0.5) == tm.CHANDELIER_MULTIPLIER_BY_REGIME["alta"]


def test_chandelier_multiplier_widens_from_baja_to_alta():
    # The literature-consistent ordering the brief specifies: more room in
    # higher-volatility regimes, less in calmer ones.
    assert tm.CHANDELIER_MULTIPLIER_BY_REGIME["baja"] < tm.CHANDELIER_MULTIPLIER_BY_REGIME["alta"]


def test_chandelier_multiplier_defaults_when_regime_unknown():
    assert tm.chandelier_multiplier(None, r_multiple=0.5) == tm.CHANDELIER_MULTIPLIER_DEFAULT
    assert tm.chandelier_multiplier("unrecognized", r_multiple=0.5) == tm.CHANDELIER_MULTIPLIER_DEFAULT


def test_chandelier_multiplier_locks_in_profit_beyond_threshold_regardless_of_regime():
    # CHANDELIER_PROFIT_LOCK_R = 1.5 (Parte 3.2/19 recalibration, was 2.0).
    assert tm.chandelier_multiplier("alta", r_multiple=1.6) == tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK


def test_chandelier_multiplier_uses_regime_below_profit_lock_threshold():
    assert tm.chandelier_multiplier("alta", r_multiple=1.0) == tm.CHANDELIER_MULTIPLIER_BY_REGIME["alta"]


# --- update_trailing_stop: must never lower --------------------------------


def test_trailing_stop_never_lowers():
    assert tm.update_trailing_stop(current_stop=100.0, candidate=95.0) == pytest.approx(100.0)


def test_trailing_stop_raises_when_candidate_is_higher():
    assert tm.update_trailing_stop(current_stop=100.0, candidate=105.0) == pytest.approx(105.0)


def test_trailing_stop_adopts_candidate_when_no_current_stop():
    assert tm.update_trailing_stop(current_stop=None, candidate=95.0) == pytest.approx(95.0)


def test_trailing_stop_keeps_current_when_no_candidate():
    assert tm.update_trailing_stop(current_stop=100.0, candidate=None) == pytest.approx(100.0)


def test_trailing_stop_none_when_neither_available():
    assert tm.update_trailing_stop(current_stop=None, candidate=None) is None


def test_trailing_stop_discards_a_corrupt_current_stop_above_price():
    # Tercera auditoría, Bloque A-1: a stop already persisted above the
    # current price (from before the entry-bounded Chandelier fix existed)
    # must not win max(current_stop, candidate) forever just because it's
    # numerically larger - it's an impossible state for an open long
    # position, not a legitimately "locked in" one.
    assert tm.update_trailing_stop(current_stop=263.0, candidate=90.0, price=95.0) == pytest.approx(90.0)


def test_trailing_stop_corrupt_current_stop_with_no_candidate_yet_returns_none():
    # No valid replacement available this bar either - honest "no valid stop
    # right now" beats silently keeping the impossible 263.0.
    assert tm.update_trailing_stop(current_stop=263.0, candidate=None, price=95.0) is None


def test_trailing_stop_price_guard_is_a_no_op_when_current_stop_is_legitimate():
    # A current_stop below price is untouched by the guard - same result as
    # before this fix for every already-healthy position.
    assert tm.update_trailing_stop(current_stop=90.0, candidate=85.0, price=95.0) == pytest.approx(90.0)


# --- compute_trailing_stop: end-to-end --------------------------------------


def test_compute_trailing_stop_never_lowers_even_when_the_candidate_is_lower():
    # CHANDELIER_WINDOW = 10 (Parte 3.2/19 recalibration, was 22) - a clean
    # 10-bar series so the window high is unambiguous.
    high = pd.Series([100.0] * 5 + [110.0, 108.0, 106.0, 104.0, 102.0])  # window high = 110
    atr14 = pd.Series([2.0] * 10)
    result = tm.compute_trailing_stop(high, atr14, current_stop=107.0, r_multiple=0.5, vol_regime="normal")
    # candidate = 110 - 2.0*2 = 106.0 < current_stop 107.0 -> stays put
    assert result.stop == pytest.approx(107.0)
    assert result.multiplier == pytest.approx(2.0)
    # El Chandelier no gobierna esta evaluación (perdió el max()) - sin texto
    # de anclaje nuevo, para que update_trailing no pise el que ya hay.
    assert result.basis is None


def test_compute_trailing_stop_raises_when_the_candidate_is_higher():
    high = pd.Series([100.0] * 5 + [140.0, 138.0, 136.0, 134.0, 132.0])
    atr14 = pd.Series([2.0] * 10)
    result = tm.compute_trailing_stop(high, atr14, current_stop=125.0, r_multiple=0.5, vol_regime="normal")
    # candidate = 140 - 2.0*2 = 136.0 > 125.0 -> raises to 136.0
    assert result.stop == pytest.approx(136.0)
    # El Chandelier sí gobierna aquí (ganó el max()) - se anota como el
    # anclaje del stop actual, para que trade_plan_service persista *por qué*
    # el stop está donde está, no solo el número.
    assert result.basis == "Chandelier 2.00x ATR desde el máximo de 10 sesiones"


def test_compute_trailing_stop_uses_the_tighter_profit_lock_multiplier_beyond_threshold():
    high = pd.Series([100.0] * 5 + [140.0, 138.0, 136.0, 134.0, 132.0])
    atr14 = pd.Series([2.0] * 10)
    # r_multiple 1.6 > CHANDELIER_PROFIT_LOCK_R (1.5) - locked regardless of "alta" regime.
    result = tm.compute_trailing_stop(high, atr14, current_stop=None, r_multiple=1.6, vol_regime="alta")
    # multiplier locked at 1.5 despite "alta" regime -> candidate = 140 - 1.5*2 = 137.0
    assert result.multiplier == pytest.approx(tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK)
    assert result.stop == pytest.approx(137.0)


def test_compute_trailing_stop_never_places_the_stop_at_or_above_the_current_price():
    # A high reached before the position's own entry (e.g. the caller failed
    # to bound `high` to the entry date) can produce a candidate above where
    # the position is trading right now - a stop a long position has
    # already been "stopped out of" by construction is never valid, and
    # since a stop only ever moves up, an unguarded bad candidate would be
    # permanent. `price` is the one number that catches it regardless of how
    # `high` was sliced upstream.
    high = pd.Series([130.0, 102.0, 101.0])  # a pre-entry high of 130, then the real (lower) trade
    atr14 = pd.Series([2.0] * 3)
    result = tm.compute_trailing_stop(
        high, atr14, current_stop=95.0, r_multiple=0.3, vol_regime="normal", price=101.0
    )
    # Unguarded candidate would be 130 - 3*2 = 124.0, above the 101.0 price -
    # discarded, current_stop (95.0, itself already below price) stands.
    assert result.stop == pytest.approx(95.0)


def test_compute_trailing_stop_self_heals_a_corrupt_stop_persisted_before_the_fix():
    # Reproduces the brief's exact case: a plan whose current_stop (263.0)
    # was left above price by the pre-fix Chandelier bug. Three consecutive
    # evaluations, same shape as the real evaluate-and-persist loop in
    # portfolio_risk_service.py, must recover a valid (below-price) stop
    # immediately on the first one - not stay wedged at 263.0 forever.
    high = pd.Series([90.0] * 17 + [96.0, 95.0, 94.0, 93.0, 92.0])  # window high = 96
    atr14 = pd.Series([1.0] * 22)
    current_stop = 263.0
    for _ in range(3):
        result = tm.compute_trailing_stop(
            high, atr14, current_stop=current_stop, r_multiple=0.2, vol_regime="normal", price=95.0
        )
        assert result.stop is not None
        assert result.stop < 95.0
        current_stop = result.stop


def test_compute_trailing_stop_price_guard_does_not_block_a_valid_candidate_below_price():
    high = pd.Series([100.0] * 5 + [140.0, 138.0, 136.0, 134.0, 132.0])
    atr14 = pd.Series([2.0] * 10)
    result = tm.compute_trailing_stop(
        high, atr14, current_stop=125.0, r_multiple=0.5, vol_regime="normal", price=140.0
    )
    # candidate = 140 - 2.0*2 = 136.0, safely below the 140.0 price -> raises
    # normally, same result as the pre-existing no-guard "raises" test.
    assert result.stop == pytest.approx(136.0)


# --- max_shares_for_position_risk -------------------------------------------


def test_max_shares_for_position_risk_basic():
    # 1% of 100,000 = 1,000 max risk; risk/share = 50-45 = 5 -> 200 shares
    assert tm.max_shares_for_position_risk(100_000.0, 50.0, 45.0) == pytest.approx(200.0)


def test_max_shares_for_position_risk_none_when_stop_at_or_above_entry():
    assert tm.max_shares_for_position_risk(100_000.0, 50.0, 50.0) is None
    assert tm.max_shares_for_position_risk(100_000.0, 50.0, 52.0) is None


def test_max_shares_for_position_risk_none_with_no_capital():
    assert tm.max_shares_for_position_risk(0.0, 50.0, 45.0) is None


def test_max_shares_for_position_risk_respects_custom_risk_pct():
    assert tm.max_shares_for_position_risk(100_000.0, 50.0, 45.0, max_risk_pct=0.02) == pytest.approx(400.0)


# --- compute_scaled_exit_plan ------------------------------------------------


def test_scaled_exit_none_below_1r():
    plan = tm.compute_scaled_exit_plan(
        r_multiple=0.5, quantity_held=100.0, initial_quantity=100.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE
    assert plan.shares_to_sell == pytest.approx(0.0)


def test_scaled_exit_sell_at_1r_with_full_position_still_held():
    # SCALE_OUT_1R_FRACTION = 0.33 (Parte 8/19) - 100 shares makes the math clean.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=1.2, quantity_held=100.0, initial_quantity=100.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.SELL_AT_1R
    assert plan.shares_to_sell == pytest.approx(33.0)
    assert plan.shares_remaining_after == pytest.approx(67.0)
    # Parte 8: break-even INCLUDES the round-trip cost (TRANSACTION_COST_PCT
    # 0.1%/side -> 0.2% round trip), not the bare entry price.
    assert plan.suggested_new_stop == pytest.approx(100.2)
    assert "33" in plan.description and "100" in plan.description


def test_scaled_exit_no_repeat_1r_suggestion_once_already_scaled():
    # Already sold ~1/3 (67 of 100 originally held) - not yet at +2R.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=1.5, quantity_held=67.0, initial_quantity=100.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_sell_at_2r_once_already_scaled_once():
    # Parte 8: the +2R stop moves to the +1R price level, not just "the
    # Chandelier trail governs from here" - initial_stop=90.0 makes 1R = 10.0,
    # so +1R = 110.0.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=2.1, quantity_held=67.0, initial_quantity=100.0, entry_price=100.0, initial_stop=90.0
    )
    assert plan.action == tm.ScaleOutAction.SELL_AT_2R
    assert plan.shares_to_sell == pytest.approx(33.0)
    assert plan.shares_remaining_after == pytest.approx(34.0)  # Parte 8's own "remaining 34%"
    assert plan.suggested_new_stop == pytest.approx(110.0)


def test_scaled_exit_sell_at_2r_without_initial_stop_leaves_suggested_stop_none():
    # Graceful degradation: no initial_stop given -> can't compute the +1R
    # price, so no stop is suggested (same as before this behavior existed),
    # rather than fabricating one.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=2.1, quantity_held=67.0, initial_quantity=100.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.SELL_AT_2R
    assert plan.suggested_new_stop is None


def test_scaled_exit_none_once_already_scaled_twice():
    # Already down to the last tranche (~34 of 100) - both milestones
    # handled, and no bars_held given so the time-stop can't fire either.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=3.0, quantity_held=34.0, initial_quantity=100.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_closes_the_last_tranche_after_the_time_stop_without_3r():
    # Parte 8/15 scenario #17: last third, 16 sessions, no +3R -> time close.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=1.8, quantity_held=34.0, initial_quantity=100.0, entry_price=100.0, bars_held=16
    )
    assert plan.action == tm.ScaleOutAction.CLOSE_LAST_TRANCHE
    assert plan.shares_to_sell == pytest.approx(34.0)
    assert plan.shares_remaining_after == pytest.approx(0.0)


def test_scaled_exit_last_tranche_not_yet_at_the_time_stop_boundary():
    # LAST_TRANCHE_TIME_STOP_BARS = 15 - exactly 15 sessions doesn't close yet.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=1.8, quantity_held=34.0, initial_quantity=100.0, entry_price=100.0, bars_held=15
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_last_tranche_time_stop_does_not_fire_once_3r_is_reached():
    plan = tm.compute_scaled_exit_plan(
        r_multiple=3.0, quantity_held=34.0, initial_quantity=100.0, entry_price=100.0, bars_held=16
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_small_position_does_not_scale_at_1r():
    # Parte 8: a position under MIN_POSITION_FOR_SCALING ($150) at open never
    # scales - a third of a $100 position is ~$33, commission eats the profit.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=1.2, quantity_held=10.0, initial_quantity=10.0, entry_price=10.0
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_small_position_exits_whole_at_2r():
    plan = tm.compute_scaled_exit_plan(
        r_multiple=2.1, quantity_held=10.0, initial_quantity=10.0, entry_price=10.0
    )
    assert plan.action == tm.ScaleOutAction.SELL_AT_2R
    assert plan.shares_to_sell == pytest.approx(10.0)
    assert plan.shares_remaining_after == pytest.approx(0.0)
    assert plan.suggested_new_stop is None


def test_scaled_exit_none_without_a_resolvable_r_multiple():
    plan = tm.compute_scaled_exit_plan(
        r_multiple=None, quantity_held=100.0, initial_quantity=100.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_none_when_position_is_already_fully_closed():
    plan = tm.compute_scaled_exit_plan(r_multiple=3.0, quantity_held=0.0, initial_quantity=30.0, entry_price=100.0)
    assert plan.action == tm.ScaleOutAction.NONE
