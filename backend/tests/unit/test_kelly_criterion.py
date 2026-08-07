import pytest

from app.services import kelly_criterion as kc


def test_kelly_fraction_binary_positive_edge():
    f = kc.kelly_fraction_binary(win_probability=0.6, reward_risk_ratio=2.0)
    assert f == pytest.approx(0.4)


def test_kelly_fraction_binary_negative_edge():
    f = kc.kelly_fraction_binary(win_probability=0.3, reward_risk_ratio=1.0)
    assert f == pytest.approx(-0.4)


def test_kelly_fraction_binary_zero_reward_risk_returns_zero():
    assert kc.kelly_fraction_binary(win_probability=0.9, reward_risk_ratio=0.0) == 0.0


def test_win_probability_from_barriers_normalizes_by_resolved_paths():
    p = kc.win_probability_from_barriers(prob_target_first=0.4, prob_stop_first=0.2)
    assert p == pytest.approx(2 / 3)


def test_win_probability_from_barriers_none_when_nothing_resolved():
    assert kc.win_probability_from_barriers(prob_target_first=0.0, prob_stop_first=0.0) is None


def test_recommend_position_size_caps_at_max_fraction_for_a_huge_edge():
    result = kc.recommend_position_size(win_probability=0.9, reward_risk_ratio=5.0)
    assert result.recommended_position_pct == pytest.approx(kc.MAX_POSITION_FRACTION)
    assert "techo" in result.rationale


def test_recommend_position_size_zero_for_negative_edge():
    result = kc.recommend_position_size(win_probability=0.2, reward_risk_ratio=1.0)
    assert result.recommended_position_pct == 0.0
    assert "negativo" in result.rationale


def test_recommend_position_size_applies_vol_regime_multiplier():
    baseline = kc.recommend_position_size(win_probability=0.55, reward_risk_ratio=2.0, vol_regime=None)
    high_vol = kc.recommend_position_size(win_probability=0.55, reward_risk_ratio=2.0, vol_regime="alta")
    assert high_vol.recommended_position_pct < baseline.recommended_position_pct
    assert high_vol.recommended_position_pct == pytest.approx(baseline.recommended_position_pct * 0.5)


def test_recommend_position_size_normal_regime_matches_baseline():
    baseline = kc.recommend_position_size(win_probability=0.55, reward_risk_ratio=2.0, vol_regime=None)
    normal = kc.recommend_position_size(win_probability=0.55, reward_risk_ratio=2.0, vol_regime="normal")
    assert normal.recommended_position_pct == pytest.approx(baseline.recommended_position_pct)


def test_growth_rate_retained_matches_half_kelly_formula():
    result = kc.recommend_position_size(win_probability=0.55, reward_risk_ratio=2.0)
    assert result.growth_rate_retained_pct == pytest.approx(0.5 * 1.5)
