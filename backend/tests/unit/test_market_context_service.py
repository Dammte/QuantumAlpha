import pytest

from app.services.macro_data_service import MacroSnapshot
from app.services.market_context_service import (
    REGIME_AVOID,
    REGIME_CAUTION,
    REGIME_FAVORABLE,
    FearGreed,
    Liquidity,
    VixSnapshot,
    _normalize,
    assess_market_regime,
    fear_greed_label,
)


def test_normalize_clamps_to_0_and_100():
    assert _normalize(-1.0, -0.08, 0.08) == pytest.approx(0.0)
    assert _normalize(1.0, -0.08, 0.08) == pytest.approx(100.0)


def test_normalize_midpoint_is_50():
    assert _normalize(0.0, -0.08, 0.08) == pytest.approx(50.0)


def test_normalize_invert_flips_the_scale():
    assert _normalize(1.0, -0.08, 0.08, invert=True) == pytest.approx(0.0)
    assert _normalize(-1.0, -0.08, 0.08, invert=True) == pytest.approx(100.0)


def test_normalize_none_defaults_to_neutral():
    assert _normalize(None, -0.08, 0.08) == pytest.approx(50.0)


@pytest.mark.parametrize(
    "score,expected",
    [
        (10, "Miedo extremo"),
        (30, "Miedo"),
        (50, "Neutral"),
        (65, "Codicia"),
        (90, "Codicia extrema"),
    ],
)
def test_fear_greed_label_bands(score, expected):
    assert fear_greed_label(score) == expected


def _vix(regime: str, level: float | None = 15.0, term_structure: str | None = "contango (normal)") -> VixSnapshot:
    return VixSnapshot(level=level, sma50=level, regime=regime, term_structure=term_structure)


def _fg(score: float = 50.0) -> FearGreed:
    return FearGreed(score=score, label=fear_greed_label(score), components={})


def _liq(headwind: bool = False) -> Liquidity:
    return Liquidity(proxy_ticker="UUP", trend="neutral", headwind=headwind)


def test_regime_favorable_when_everything_calm():
    regime = assess_market_regime(_vix("normal"), _fg(50), _liq(False))
    assert regime.verdict == REGIME_FAVORABLE
    assert "insuficiente" not in regime.headline


def test_regime_avoid_when_vix_in_panic_regardless_of_everything_else():
    regime = assess_market_regime(_vix("pánico", level=32), _fg(50), _liq(False))
    assert regime.verdict == REGIME_AVOID


def test_regime_avoid_when_vix_in_crisis():
    regime = assess_market_regime(_vix("crisis", level=45), _fg(50), _liq(False))
    assert regime.verdict == REGIME_AVOID


def test_regime_caution_for_a_single_elevated_signal():
    regime = assess_market_regime(_vix("miedo elevado", level=25), _fg(50), _liq(False))
    assert regime.verdict == REGIME_CAUTION
    assert any("VIX" in r for r in regime.reasons)


def test_regime_avoid_when_multiple_corroborating_signals_stack_up_below_panic_vix():
    regime = assess_market_regime(
        _vix("miedo elevado", level=25, term_structure="backwardation (estrés)"), _fg(15), _liq(True)
    )
    assert regime.verdict == REGIME_AVOID
    assert len(regime.reasons) >= 3


def test_regime_caution_on_extreme_greed_not_just_extreme_fear():
    regime = assess_market_regime(_vix("normal"), _fg(90), _liq(False))
    assert regime.verdict == REGIME_CAUTION
    assert any("codicia" in r.lower() for r in regime.reasons)


def test_regime_reasons_never_empty():
    regime = assess_market_regime(_vix("complacencia", level=9), _fg(50), _liq(False))
    assert len(regime.reasons) > 0


def _macro(yield_curve_spread=0.3, yield_curve_inverted=False):
    return MacroSnapshot(
        yield_curve_spread=yield_curve_spread,
        yield_curve_date="2026-08-01",
        yield_curve_inverted=yield_curve_inverted,
        unemployment_rate=4.0,
        unemployment_date="2026-07-01",
        cpi_yoy_change=2.8,
        cpi_date="2026-07-01",
    )


def test_regime_unaffected_by_normal_yield_curve():
    regime = assess_market_regime(_vix("normal"), _fg(50), _liq(False), macro=_macro(0.3, False))
    assert regime.verdict == REGIME_FAVORABLE


def test_regime_caution_when_yield_curve_inverted():
    regime = assess_market_regime(_vix("normal"), _fg(50), _liq(False), macro=_macro(-0.15, True))
    assert regime.verdict == REGIME_CAUTION
    assert any("invertida" in r for r in regime.reasons)


def test_regime_avoid_when_inverted_curve_stacks_with_other_signals():
    regime = assess_market_regime(
        _vix("miedo elevado", level=25), _fg(15), _liq(False), macro=_macro(-0.3, True)
    )
    assert regime.verdict == REGIME_AVOID


def test_regime_ignores_macro_when_none_provided():
    regime = assess_market_regime(_vix("normal"), _fg(50), _liq(False), macro=None)
    assert regime.verdict == REGIME_FAVORABLE
