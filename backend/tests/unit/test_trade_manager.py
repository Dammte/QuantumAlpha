import pandas as pd
import pytest

from app.services import trade_manager as tm

# --- chandelier_stop ---------------------------------------------------------


def test_chandelier_stop_basic():
    high = pd.Series([100.0, 102.0, 105.0, 103.0, 101.0])
    atr14 = pd.Series([2.0] * 5)
    result = tm.chandelier_stop(high, atr14, multiplier=3.0, window=5)
    assert result == pytest.approx(105.0 - 3 * 2.0)


def test_chandelier_stop_none_with_insufficient_bars():
    high = pd.Series([100.0, 102.0])
    atr14 = pd.Series([2.0] * 2)
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


def test_chandelier_multiplier_locks_in_profit_beyond_2r_regardless_of_regime():
    assert tm.chandelier_multiplier("alta", r_multiple=2.5) == tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK


def test_chandelier_multiplier_uses_regime_below_2r():
    assert tm.chandelier_multiplier("alta", r_multiple=1.9) == tm.CHANDELIER_MULTIPLIER_BY_REGIME["alta"]


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


# --- compute_trailing_stop: end-to-end --------------------------------------


def test_compute_trailing_stop_never_lowers_even_when_the_candidate_is_lower():
    high = pd.Series([100.0] * 17 + [130.0, 128.0, 126.0, 124.0, 122.0])  # 22 bars, window high = 130
    atr14 = pd.Series([2.0] * 22)
    result = tm.compute_trailing_stop(high, atr14, current_stop=125.0, r_multiple=0.5, vol_regime="normal")
    # candidate = 130 - 3.0*2 = 124.0 < current_stop 125.0 -> stays put
    assert result.stop == pytest.approx(125.0)
    assert result.multiplier == pytest.approx(3.0)


def test_compute_trailing_stop_raises_when_the_candidate_is_higher():
    high = pd.Series([100.0] * 17 + [140.0, 138.0, 136.0, 134.0, 132.0])
    atr14 = pd.Series([2.0] * 22)
    result = tm.compute_trailing_stop(high, atr14, current_stop=125.0, r_multiple=0.5, vol_regime="normal")
    # candidate = 140 - 3*2 = 134.0 > 125.0 -> raises to 134.0
    assert result.stop == pytest.approx(134.0)


def test_compute_trailing_stop_uses_the_tighter_profit_lock_multiplier_beyond_2r():
    high = pd.Series([100.0] * 17 + [140.0, 138.0, 136.0, 134.0, 132.0])
    atr14 = pd.Series([2.0] * 22)
    result = tm.compute_trailing_stop(high, atr14, current_stop=None, r_multiple=2.5, vol_regime="alta")
    # multiplier locked at 2.0 despite "alta" regime -> candidate = 140 - 2*2 = 136.0
    assert result.multiplier == pytest.approx(tm.CHANDELIER_MULTIPLIER_PROFIT_LOCK)
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
        r_multiple=0.5, quantity_held=30.0, initial_quantity=30.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE
    assert plan.shares_to_sell == pytest.approx(0.0)


def test_scaled_exit_sell_at_1r_with_full_position_still_held():
    plan = tm.compute_scaled_exit_plan(
        r_multiple=1.2, quantity_held=30.0, initial_quantity=30.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.SELL_AT_1R
    assert plan.shares_to_sell == pytest.approx(10.0)
    assert plan.shares_remaining_after == pytest.approx(20.0)
    assert plan.suggested_new_stop == pytest.approx(100.0)  # break-even
    assert "10" in plan.description and "30" in plan.description


def test_scaled_exit_no_repeat_1r_suggestion_once_already_scaled():
    # Already sold ~1/3 (20 of 30 originally held) - not yet at +2R.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=1.5, quantity_held=20.0, initial_quantity=30.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_sell_at_2r_once_already_scaled_once():
    plan = tm.compute_scaled_exit_plan(
        r_multiple=2.1, quantity_held=20.0, initial_quantity=30.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.SELL_AT_2R
    assert plan.shares_to_sell == pytest.approx(10.0)
    assert plan.shares_remaining_after == pytest.approx(10.0)
    assert plan.suggested_new_stop is None  # the Chandelier trail governs from here, not a fixed level


def test_scaled_exit_none_once_already_scaled_twice():
    # Already down to ~1/3 remaining (10 of 30) - both milestones handled.
    plan = tm.compute_scaled_exit_plan(
        r_multiple=3.0, quantity_held=10.0, initial_quantity=30.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_none_without_a_resolvable_r_multiple():
    plan = tm.compute_scaled_exit_plan(
        r_multiple=None, quantity_held=30.0, initial_quantity=30.0, entry_price=100.0
    )
    assert plan.action == tm.ScaleOutAction.NONE


def test_scaled_exit_none_when_position_is_already_fully_closed():
    plan = tm.compute_scaled_exit_plan(r_multiple=3.0, quantity_held=0.0, initial_quantity=30.0, entry_price=100.0)
    assert plan.action == tm.ScaleOutAction.NONE
