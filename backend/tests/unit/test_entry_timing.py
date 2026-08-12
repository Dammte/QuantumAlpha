from app.services.entry_timing import EXTENDED, LATE, OPTIMAL, VALID, assess_entry_timing
from app.services.technical_analysis import PriceLevel, TrendState


def _support(distance_pct: float) -> PriceLevel:
    return PriceLevel(price=100.0, kind="support", strength=2, distance_pct=distance_pct)


def test_none_atr_multiple_returns_none() -> None:
    assert assess_entry_timing(None, None, TrendState.UPTREND) is None


def test_beyond_extended_threshold_is_extended() -> None:
    result = assess_entry_timing(4.5, None, TrendState.UPTREND)
    assert result.status == EXTENDED


def test_between_late_and_extended_threshold_is_late() -> None:
    result = assess_entry_timing(3.0, None, TrendState.UPTREND)
    assert result.status == LATE


def test_near_support_in_uptrend_is_optimal_even_with_moderate_atr() -> None:
    result = assess_entry_timing(2.0, _support(0.01), TrendState.UPTREND)
    assert result.status == OPTIMAL


def test_near_support_in_downtrend_is_not_optimal() -> None:
    """Mirrors recommendation_engine.py's own "near support" gating: a pullback
    only reads as low-risk in a trend that's actually still up."""
    result = assess_entry_timing(2.0, _support(0.01), TrendState.DOWNTREND)
    assert result.status == VALID


def test_very_low_atr_multiple_is_optimal_regardless_of_support() -> None:
    result = assess_entry_timing(0.5, None, TrendState.SIDEWAYS)
    assert result.status == OPTIMAL


def test_moderate_atr_multiple_away_from_support_is_valid() -> None:
    result = assess_entry_timing(2.0, None, TrendState.UPTREND)
    assert result.status == VALID


def test_labels_and_descriptions_are_populated_for_every_status() -> None:
    for atr in (0.5, 2.0, 3.0, 4.5):
        result = assess_entry_timing(atr, None, TrendState.UPTREND)
        assert result.label
        assert result.description
        assert result.atr_multiple == atr
