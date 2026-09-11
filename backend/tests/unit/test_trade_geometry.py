import pytest

from app.services import trade_geometry as tg
from app.services.technical_analysis import PriceLevel

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
