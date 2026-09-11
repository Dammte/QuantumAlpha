import pytest

from app.services import levels_engine as le
from app.services.technical_analysis import PriceLevel, Stage, TrendState

# A support at exactly the 3% pullback-proximity boundary and no resistance -
# relies on trade_geometry's fixed 2:1 fallback target (no resistance to
# second-guess it), so risk_reward is deterministically 2.0 regardless of the
# exact stop math - every condition passes by construction.


def _passing_kwargs() -> dict:
    return dict(
        price=100.0,
        trend=TrendState.UPTREND,
        stage=Stage.STAGE_2,
        rsi14=50.0,
        adx14=30.0,
        plus_di=25.0,
        minus_di=10.0,
        atr14=2.0,
        atr_multiple=1.0,
        nearest_support=PriceLevel(price=97.0, kind="support", strength=2, distance_pct=-0.03),
        nearest_resistance=None,
        obv_divergence=None,
        fast_pair_bearish_signal=None,
    )


def test_all_conditions_pass_is_a_clean_gate():
    result = le.evaluate_gate(**_passing_kwargs())
    assert result.passes is True
    assert all(c.passed for c in result.conditions)
    assert result.entry_trigger is not None
    assert result.entry_trigger.trigger_type == "pullback_bounce"
    assert result.stop_and_target.risk_reward == pytest.approx(2.0)


def test_gate_fails_when_trend_is_down_and_not_stage2():
    result = le.evaluate_gate(**{**_passing_kwargs(), "trend": TrendState.DOWNTREND, "stage": None})
    assert result.passes is False
    trend_condition = next(c for c in result.conditions if "Tendencia" in c.label)
    assert trend_condition.passed is False


def test_gate_passes_on_stage2_alone_even_if_trend_reads_sideways():
    result = le.evaluate_gate(**{**_passing_kwargs(), "trend": TrendState.SIDEWAYS, "stage": Stage.STAGE_2})
    trend_condition = next(c for c in result.conditions if "Tendencia" in c.label)
    assert trend_condition.passed is True


def test_gate_fails_on_parabolic_extension():
    result = le.evaluate_gate(**{**_passing_kwargs(), "atr_multiple": 5.0})
    assert result.passes is False
    condition = next(c for c in result.conditions if "parabólica" in c.label)
    assert condition.passed is False


def test_gate_fails_on_overbought_outside_strong_trend():
    result = le.evaluate_gate(**{**_passing_kwargs(), "rsi14": 85.0, "adx14": 15.0})
    assert result.passes is False
    condition = next(c for c in result.conditions if "sobrecompra" in c.label)
    assert condition.passed is False


def test_gate_tolerates_high_rsi_inside_a_strong_confirmed_uptrend():
    # Same "don't fight the trend factors" reasoning recommendation_engine.py
    # used to encode - RSI pinned above 80 for weeks is normal in a genuinely
    # strong, ADX-confirmed uptrend, not a warning sign.
    result = le.evaluate_gate(**{**_passing_kwargs(), "rsi14": 85.0})
    condition = next(c for c in result.conditions if "sobrecompra" in c.label)
    assert condition.passed is True


def test_gate_fails_on_bearish_obv_divergence():
    result = le.evaluate_gate(**{**_passing_kwargs(), "obv_divergence": "bearish"})
    assert result.passes is False
    condition = next(c for c in result.conditions if "OBV" in c.label)
    assert condition.passed is False


def test_gate_fails_on_fast_pair_bearish_veto():
    result = le.evaluate_gate(**{**_passing_kwargs(), "fast_pair_bearish_signal": "death cross EMA21/55"})
    assert result.passes is False
    condition = next(c for c in result.conditions if "par rápido" in c.label)
    assert condition.passed is False


def test_gate_fails_when_atr_missing_so_no_stop_can_be_computed():
    result = le.evaluate_gate(**{**_passing_kwargs(), "atr14": None})
    assert result.passes is False
    assert result.stop_and_target.stop_loss is None
    condition = next(c for c in result.conditions if "beneficio:riesgo" in c.label)
    assert condition.passed is False


def test_gate_fails_when_a_nearby_resistance_caps_reward_risk_below_minimum():
    # Resistance close enough to qualify as the target (>=1.0 reward:risk,
    # trade_geometry's own bar) but not enough to clear this gate's stricter
    # 1.5 minimum - risk ~3.97 (support-based stop), reward 4.5 -> ~1.13.
    kwargs = {
        **_passing_kwargs(),
        "nearest_resistance": PriceLevel(price=104.5, kind="resistance", strength=1, distance_pct=0.045),
    }
    result = le.evaluate_gate(**kwargs)
    assert result.passes is False
    assert result.stop_and_target.take_profit_method == "resistencia más cercana"
    assert result.stop_and_target.risk_reward < le.MIN_REWARD_RISK
    condition = next(c for c in result.conditions if "beneficio:riesgo" in c.label)
    assert condition.passed is False


def test_gate_result_lists_every_condition_even_when_several_fail_at_once():
    kwargs = {**_passing_kwargs(), "atr_multiple": 5.0, "obv_divergence": "bearish"}
    result = le.evaluate_gate(**kwargs)
    assert len(result.conditions) == 6
    failed_labels = {c.label for c in result.conditions if not c.passed}
    assert any("parabólica" in label for label in failed_labels)
    assert any("OBV" in label for label in failed_labels)


def test_entry_trigger_none_does_not_by_itself_fail_the_gate():
    # A clean setup with no support/resistance close enough to define a
    # watchable level yet is still a passing gate - "no imminent trigger"
    # and "bad entry" are different questions (see module docstring).
    result = le.evaluate_gate(**{**_passing_kwargs(), "nearest_support": None, "nearest_resistance": None})
    assert result.entry_trigger is None
    assert result.passes is True
