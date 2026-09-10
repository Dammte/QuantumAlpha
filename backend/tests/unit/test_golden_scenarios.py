"""Cuarta auditoría independiente, Bloques C/D: a golden-scenario test suite
for the recommendation engine - hand-built, textbook-shaped price series with
an obvious, human-verifiable correct answer, run end to end through the real
pipeline (`technical_analysis.py` indicators → `recommendation_engine.build_recommendation`,
including the new fast-pair veto from Bloque B), not through a single
isolated function the way the rest of `tests/unit/` deliberately does.

Distinct from every other test file in this project on purpose: everywhere
else, a test isolates ONE function with hand-picked scalar inputs (the
established, correct style for verifying a single piece of logic in
isolation - see `test_recommendation_engine.py`/`test_technical_analysis.py`).
This file instead asks "does the whole pipeline agree with what a human
technical trader would call this chart, end to end?" - a regression net for
the system's actual, composed behavior, not a substitute for the
function-level tests. `exit_engine.py` (the sell-side engine) already has
this kind of end-to-end scenario coverage in its own 47-test file
(`test_exit_engine.py`) - this file is the buy-side counterpart that never
existed.

Assertions here are deliberately high-level (verdict, or "score cleared
BUY_THRESHOLD") rather than exact scores - the whole point of a golden
scenario is "would a human call this a buy", not "does this series produce
score == 7 forever", which would make the suite brittle against a future,
legitimate factor recalibration that doesn't change the verdict a human
would expect."""

import numpy as np
import pandas as pd

from app.services import technical_analysis as ta
from app.services.recommendation_engine import Recommendation, build_recommendation


def _last(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _recommendation_from_series(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
    rs_rating: int | None = None,
    apply_veto: bool = True,
) -> Recommendation:
    """Reproduces the same indicator-derivation pipeline
    `ticker_analysis_service._confirmed_recommendation`/`compute_core_signals`
    use before calling the real `build_recommendation`."""
    high = high if high is not None else close * 1.01
    low = low if low is not None else close * 0.99
    volume = volume if volume is not None else pd.Series([1_000_000.0] * len(close), index=close.index)

    price = float(close.iloc[-1])
    sma20_s, sma50_s = ta.sma(close, 20), ta.sma(close, 50)
    sma150_s, sma200_s = ta.sma(close, 150), ta.sma(close, 200)
    sma20, sma50, sma150, sma200 = _last(sma20_s), _last(sma50_s), _last(sma150_s), _last(sma200_s)
    trend = ta.classify_trend(price, sma20, sma50, sma200)

    stage, ma_cross = None, None
    if len(close) >= 200:
        stage = ta.classify_stage(price, sma150_s)
        ma_cross = ta.detect_recent_cross(sma50_s, sma200_s, lookback=5)

    adx_s = ta.adx(high, low, close)
    plus_di_s, minus_di_s = ta.dmi(high, low, close)
    atr_s = ta.atr(high, low, close)
    atr_multiple = ta.atr_multiple_from_sma(close, high, low)

    sma200_trending_up = ta.sma_slope_positive(sma200_s)
    price_52w_low = ta.rolling_extreme_price(close, 252, "low")
    price_52w_high = ta.rolling_extreme_price(close, 252, "high")
    criteria = ta.minervini_checklist(
        price, sma50, sma150, sma200, sma200_trending_up, price_52w_low, price_52w_high, rs_rating
    )
    minervini_pass = all(criteria.values())
    minervini_range_confirmed = (
        criteria["price_25pct_above_52w_low"] and criteria["price_within_25pct_of_52w_high"]
    )

    levels = ta.support_resistance_levels(high, low, close)
    supports = [lv for lv in levels if lv.kind == "support"]
    resistances = [lv for lv in levels if lv.kind == "resistance"]
    nearest_support = min(supports, key=lambda lv: abs(lv.distance_pct)) if supports else None
    nearest_resistance = min(resistances, key=lambda lv: abs(lv.distance_pct)) if resistances else None

    obv_div = ta.obv_divergence(close, volume)
    fast_pair_veto = ta.detect_fast_pair_bearish_veto(close) if apply_veto else None

    return build_recommendation(
        price=price,
        trend=trend,
        stage=stage,
        ma_cross=ma_cross,
        rsi14=_last(ta.rsi(close)),
        adx14=_last(adx_s),
        plus_di=_last(plus_di_s),
        minus_di=_last(minus_di_s),
        atr14=_last(atr_s),
        atr_multiple=atr_multiple,
        rs_rating=rs_rating,
        minervini_pass=minervini_pass,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        minervini_range_confirmed=minervini_range_confirmed,
        obv_divergence=obv_div,
        fast_pair_bearish_signal=fast_pair_veto,
    )


def test_golden_clean_stage2_breakout_with_volume_recommends_buy():
    # A flat, multi-month base (a real consolidation, not a straight line -
    # Minervini/Weinstein both describe Stage 2 as breaking out *of* a base,
    # not appearing from nowhere) followed by a clean breakout on rising
    # volume - the textbook Stage 2 entry.
    n_base = 150
    base = 100 + np.sin(np.linspace(0, 6 * np.pi, n_base)) * 2
    n_breakout = 150
    breakout = base[-1] + np.arange(1, n_breakout + 1) * 0.6
    close = pd.Series(np.concatenate([base, breakout]))
    volume = pd.Series([1_000_000.0] * n_base + [3_000_000.0] * n_breakout)

    rec = _recommendation_from_series(close, volume=volume, rs_rating=92)

    assert rec.verdict == "comprar"
    assert rec.score >= 5  # BUY_THRESHOLD, mirrored as a literal to avoid importing implementation constants here
    assert rec.stop_loss is not None and rec.stop_loss < close.iloc[-1]
    assert rec.veto_reason is None


def test_golden_confirmed_downtrend_recommends_avoid():
    # A long, unbroken decline - Stage 4, no ambiguity about direction.
    close = pd.Series(200 - np.arange(300) * 0.3)
    rec = _recommendation_from_series(close, rs_rating=10)
    assert rec.verdict == "evitar"
    assert rec.stop_loss is None  # never offered for a non-"comprar" verdict


def test_golden_sideways_chop_recommends_wait():
    # Bounded oscillation, no net drift over the whole series - genuinely
    # directionless, the case a human would call "nothing to do here yet".
    close = pd.Series(100 + np.sin(np.linspace(0, 10 * np.pi, 300)) * 3)
    rec = _recommendation_from_series(close, rs_rating=50)
    assert rec.verdict == "esperar"


def test_golden_fast_pair_veto_suppresses_an_otherwise_clear_buy():
    # Cuarta auditoría, Bloque B (B-1.3), the golden scenario this specific
    # feature exists for: a long, genuine uptrend (real Stage 2, strong ADX,
    # a real RS leader) that has just begun a smooth, sustained recent
    # decline - still comfortably "comprar" by the slower checklist alone,
    # but the fast EMA21/55 pair is already projecting a bearish cross with
    # clean confidence. The SAME series scores identically either way
    # (`apply_veto` toggles only whether the veto is *applied*, proving this
    # changes the verdict, never the checklist's own number).
    up = 100 + np.arange(250) * 0.5
    down = up[-1] - np.arange(1, 41) * 0.2
    close = pd.Series(np.concatenate([up, down]))

    without_veto = _recommendation_from_series(close, rs_rating=90, apply_veto=False)
    with_veto = _recommendation_from_series(close, rs_rating=90, apply_veto=True)

    assert without_veto.verdict == "comprar"
    assert without_veto.veto_reason is None
    assert with_veto.verdict == "esperar"
    assert with_veto.veto_reason is not None
    assert "EMA21" in with_veto.veto_reason and "EMA55" in with_veto.veto_reason
    assert with_veto.score == without_veto.score  # the checklist's own number is untouched by the veto


def test_golden_bearish_obv_divergence_within_an_uptrend_is_visible_as_a_factor():
    # Wyckoff's "effort vs result": in the final stretch, price still creeps
    # to marginal new highs (mostly small up-days on light volume), but every
    # third day is a real down-day on heavy volume - the advance is no longer
    # backed by real net buying pressure, even though price alone still looks
    # fine. A price-only checklist can't see this by construction; OBV
    # divergence is the one factor here that isn't derived from price alone.
    n = 240
    close_base = 100 + np.arange(n) * 0.4
    tail_changes = [-0.3 if i % 3 == 2 else 0.5 for i in range(22)]
    tail_volume = [3_000_000.0 if i % 3 == 2 else 500_000.0 for i in range(22)]
    tail_close = []
    level = close_base[-1]
    for c in tail_changes:
        level += c
        tail_close.append(level)
    close = pd.Series(np.concatenate([close_base, tail_close]))
    volume = pd.Series([1_000_000.0] * n + tail_volume)

    rec = _recommendation_from_series(close, volume=volume, rs_rating=80)

    assert any(f.triggered and "OBV" in f.label and f.points < 0 for f in rec.factors)
