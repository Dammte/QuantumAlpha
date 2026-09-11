from app.services.market_context_service import (
    REGIME_AVOID,
    REGIME_CAUTION,
    REGIME_FAVORABLE,
    VixSnapshot,
    assess_market_regime,
)


def _vix(regime: str, level: float | None = 15.0, term_structure: str | None = "contango (normal)") -> VixSnapshot:
    return VixSnapshot(level=level, sma50=level, regime=regime, term_structure=term_structure)


def test_regime_favorable_when_vix_calm():
    regime = assess_market_regime(_vix("normal"))
    assert regime.verdict == REGIME_FAVORABLE
    assert "insuficiente" not in regime.headline


def test_regime_avoid_when_vix_in_panic():
    regime = assess_market_regime(_vix("pánico", level=32))
    assert regime.verdict == REGIME_AVOID


def test_regime_avoid_when_vix_in_crisis():
    regime = assess_market_regime(_vix("crisis", level=45))
    assert regime.verdict == REGIME_AVOID


def test_regime_caution_for_a_single_elevated_signal():
    regime = assess_market_regime(_vix("miedo elevado", level=25))
    assert regime.verdict == REGIME_CAUTION
    assert any("VIX" in r for r in regime.reasons)


def test_regime_avoid_when_elevated_vix_stacks_with_backwardation():
    # "miedo elevado" alone is 1 caution point (REGIME_CAUTION); backwardation
    # adds one more, reaching the >=3-point AVOID threshold only combined with
    # a regime that already contributes 2 (see _VIX_CAUTION_POINTS).
    regime = assess_market_regime(_vix("pánico", level=28, term_structure="backwardation (estrés)"))
    assert regime.verdict == REGIME_AVOID
    assert len(regime.reasons) >= 2


def test_regime_reasons_never_empty():
    regime = assess_market_regime(_vix("complacencia", level=9))
    assert len(regime.reasons) > 0


def test_regime_no_vix_data_defaults_to_favorable():
    regime = assess_market_regime(_vix("desconocido", level=None, term_structure=None))
    assert regime.verdict == REGIME_FAVORABLE
