from datetime import date

import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import registry as reg
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage


def _mtf() -> mtf.MultiTimeframeRead:
    # Registry-dispatch tests never read `ctx`'s content (that's each
    # detector's own job, tested in that detector's own module) - a minimal
    # but genuinely-typed MultiTimeframeRead, same helper pattern
    # test_exit_engine.py already uses, is enough to satisfy SetupContext's
    # own field type without a real OHLCV frame behind it.
    daily = mtf.TimeframeRead(
        timeframe="daily", trend=ta.TrendState.SIDEWAYS, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )
    return mtf.MultiTimeframeRead(
        weekly=None, daily=daily, intraday=None, alignment="transitioning", alignment_score=0.0, conflicts=[]
    )


def _ctx(**overrides) -> SetupContext:
    empty = pd.Series(dtype=float)
    defaults = dict(
        ticker="XYZ",
        region="us",
        trade_date=date(2026, 9, 17),
        close=empty,
        high=empty,
        low=empty,
        volume=empty,
        open_=empty,
        weekly_close=None,
        weekly_high=None,
        weekly_low=None,
        weekly_volume=None,
        atr_series=empty,
        atr14=None,
        ema21=None,
        ema55=None,
        sma20=None,
        sma50=None,
        sma150=None,
        sma200=None,
        levels=[],
        multi_timeframe=_mtf(),
        trend=ta.TrendState.SIDEWAYS,
        weekly_stage=None,
        relative_volume=None,
        rs_percentile=None,
        sector_rs_percentile=None,
        mansfield_rs_series=None,
    )
    defaults.update(overrides)
    return SetupContext(**defaults)


def _match(name: str = "dummy") -> SetupMatch:
    return SetupMatch(
        family=SetupFamily.STAGE_TRANSITION,
        name=name,
        label_es="Prueba",
        stage=SetupStage.READY,
        bars_in_stage=1,
        timeframe="daily",
        trigger_price=100.0,
        trigger_condition="cierre > 100",
        invalidation_price=90.0,
        invalidation_condition="cierre < 90",
        evidence={},
        narrative_es="Narrativa de prueba.",
        confidence=SetupConfidence.UNVALIDATED,
    )


def test_detect_all_with_no_registered_detectors_returns_empty(monkeypatch):
    monkeypatch.setattr(reg, "SETUP_DETECTORS", [])
    assert reg.detect_all(_ctx()) == []


def test_detect_all_collects_matches_from_every_registered_detector(monkeypatch):
    def detector_a(ctx: SetupContext) -> list[SetupMatch]:
        return [_match("a")]

    def detector_b(ctx: SetupContext) -> list[SetupMatch]:
        return [_match("b1"), _match("b2")]

    monkeypatch.setattr(reg, "SETUP_DETECTORS", [detector_a, detector_b])

    matches = reg.detect_all(_ctx())

    assert [m.name for m in matches] == ["a", "b1", "b2"]


def test_detect_all_isolates_a_detector_that_raises(monkeypatch):
    """The exact isolation `PortfolioRiskService._safe_assess_position_risk`
    already establishes for a whole ticker, one level further down: a single
    family's edge case (e.g. VCP on a ticker with too little history) must
    never take another family's genuinely valid match down with it."""

    def flaky_detector(ctx: SetupContext) -> list[SetupMatch]:
        raise ValueError("simulated detector bug")

    def healthy_detector(ctx: SetupContext) -> list[SetupMatch]:
        return [_match("healthy")]

    monkeypatch.setattr(reg, "SETUP_DETECTORS", [flaky_detector, healthy_detector])

    matches = reg.detect_all(_ctx())

    assert [m.name for m in matches] == ["healthy"]


def test_detect_all_never_orders_or_dedupes(monkeypatch):
    # detect_all's own contract (see its docstring): choosing between
    # matches is arbitration.py's job, not the registry's - a detector
    # returning duplicate-looking matches must pass through unchanged.
    def detector(ctx: SetupContext) -> list[SetupMatch]:
        return [_match("same"), _match("same")]

    monkeypatch.setattr(reg, "SETUP_DETECTORS", [detector])

    assert [m.name for m in reg.detect_all(_ctx())] == ["same", "same"]


def test_setup_detectors_only_contains_families_with_their_own_detector_and_tests():
    # A family only joins SETUP_DETECTORS in its own phase, once its
    # detector/tests/replay all exist together (see registry.py's own
    # "regla de admisión" docstring).
    from app.services.setups import ma_cross, stage_transition

    assert reg.SETUP_DETECTORS == [stage_transition.detect, ma_cross.detect]
