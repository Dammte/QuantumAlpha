"""Reconstruction (2026-09), Fase 9: golden-scenario coverage for the gate
that actually decides live now (`levels_engine.evaluate_gate`) - the same
methodology `test_golden_scenarios.py` already established for the retired
checklist (`recommendation_engine.build_recommendation`, still measured by
`scripts/factor_ablation_study.py` but no longer called from any live path -
see that module's own docstring), reapplied to what IS live. A separate
file, not a rewrite of that one: `test_golden_scenarios.py` still correctly
documents the retired checklist's own end-to-end behavior and stays as
regression coverage for it.

Same "hand-built, textbook-shaped, human-verifiable" philosophy: real
indicator series run end to end through `evaluate_gate`, asserting
`gate.passes` and *which* condition(s) failed - never internal scalar
values a legitimate future recalibration could change without changing
what a human would call the chart.

**Known, documented limitation, same one `levels_engine.replay_gate_at`
already discloses**: every scenario here passes `nearest_support=
nearest_resistance=None` (no point-in-time support/resistance scan), so
`compute_stop_and_target` always falls back to the fixed ATR-stop/2:1-target
formula and the sixth gate condition (reward:risk >= 1.5) is always `True`
here - not a scenario this file can exercise. Genuinely untested anywhere
in this suite: an entry whose nearest resistance offers *some* reward but
not enough (a resistance-based reward:risk between 1.0 and 1.5 - below that,
`compute_stop_and_target` itself falls back to the fixed 2:1 target instead
of using the resistance, which trivially clears 1.5 again). A real
`support_resistance_levels` pivot fixture landing in that exact narrow
window wasn't built for this pass - `test_portfolio_risk_service.py`'s own
near-resistance fixture (`test_strong_setup_near_resistance_is_add_candidate_not_watch`)
exercises the *passing* case, not this one."""

import numpy as np
import pandas as pd

from app.services import technical_analysis as ta
from app.services.levels_engine import GateResult, evaluate_gate


def _last(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _gate_from_series(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> GateResult:
    """Reproduces the same indicator-derivation pipeline
    `ticker_analysis_service.compute_core_signals` uses before calling the
    real `evaluate_gate` - same role `test_golden_scenarios.py`'s own
    `_recommendation_from_series` plays for the retired checklist."""
    high = high if high is not None else close * 1.01
    low = low if low is not None else close * 0.99
    volume = volume if volume is not None else pd.Series([1_000_000.0] * len(close), index=close.index)

    price = float(close.iloc[-1])
    sma20_s, sma50_s = ta.sma(close, 20), ta.sma(close, 50)
    sma150_s, sma200_s = ta.sma(close, 150), ta.sma(close, 200)
    sma20, sma50, sma200 = _last(sma20_s), _last(sma50_s), _last(sma200_s)
    trend = ta.classify_trend(price, sma20, sma50, sma200)
    stage = ta.classify_stage(price, sma150_s) if len(close) >= 200 else None

    adx_s = ta.adx(high, low, close)
    plus_di_s, minus_di_s = ta.dmi(high, low, close)
    atr_s = ta.atr(high, low, close)
    atr14 = _last(atr_s)
    atr_multiple = (price - sma50) / atr14 if atr14 else None

    obv_div = ta.obv_divergence(close, volume)
    fast_pair_veto = ta.detect_fast_pair_bearish_veto(close)

    return evaluate_gate(
        price=price,
        trend=trend,
        stage=stage,
        rsi14=_last(ta.rsi(close)),
        adx14=_last(adx_s),
        plus_di=_last(plus_di_s),
        minus_di=_last(minus_di_s),
        atr14=atr14,
        atr_multiple=atr_multiple,
        nearest_support=None,
        nearest_resistance=None,
        obv_divergence=obv_div,
        fast_pair_bearish_signal=fast_pair_veto,
    )


def _condition(gate: GateResult, label_substring: str):
    return next(c for c in gate.conditions if label_substring in c.label)


def test_golden_clean_stage2_breakout_with_volume_passes_the_gate():
    # Same shape as test_golden_scenarios.py's own breakout fixture (a real
    # multi-month base, not a straight line, then a clean breakout on rising
    # volume) - mild noise on the breakout leg (rng.normal(...).cumsum()) so
    # a smooth ramp doesn't read as "parabolic" purely from having a flat
    # ATR under it (a real pitfall found while building this file: the
    # original test's un-noised 0.6/day breakout leg tripped
    # gate_not_parabolic even with zero actual spike).
    rng = np.random.default_rng(3)
    n_base = 150
    base = 100 + np.sin(np.linspace(0, 6 * np.pi, n_base)) * 2
    n_breakout = 150
    breakout = base[-1] + np.arange(1, n_breakout + 1) * 0.4 + rng.normal(0, 1.0, n_breakout).cumsum() * 0.15
    close = pd.Series(np.concatenate([base, breakout]))
    volume = pd.Series([1_000_000.0] * n_base + [3_000_000.0] * n_breakout)

    gate = _gate_from_series(close, volume=volume)

    assert gate.passes
    assert all(c.passed for c in gate.conditions)
    assert gate.stop_and_target.stop_loss < close.iloc[-1]


def test_golden_confirmed_downtrend_fails_the_gate_on_trend():
    # A long, unbroken decline - Stage 4, no ambiguity about direction.
    close = pd.Series(200 - np.arange(300) * 0.3)
    gate = _gate_from_series(close)
    assert not _condition(gate, "Tendencia").passed
    assert not gate.passes


def test_golden_sideways_chop_fails_the_gate_on_trend():
    # Bounded oscillation, no net drift over the whole series - genuinely
    # directionless, the case a human would call "nothing to do here yet".
    close = pd.Series(100 + np.sin(np.linspace(0, 10 * np.pi, 300)) * 3)
    gate = _gate_from_series(close)
    assert not _condition(gate, "Tendencia").passed
    assert not gate.passes


def test_golden_fast_pair_veto_fails_the_gate_even_with_a_confirmed_uptrend():
    # Same fixture as test_golden_scenarios.py's own veto scenario: a long,
    # genuine uptrend that has just begun a smooth, sustained recent decline
    # - still a confirmed uptrend by the slower trend/stage read, but the
    # fast EMA21/55 pair is already projecting a bearish cross with clean
    # confidence. Isolates cleanly: every other condition still passes.
    up = 100 + np.arange(250) * 0.5
    down = up[-1] - np.arange(1, 41) * 0.2
    close = pd.Series(np.concatenate([up, down]))

    gate = _gate_from_series(close)

    assert _condition(gate, "Tendencia").passed
    assert not _condition(gate, "par rápido").passed
    assert not gate.passes
    other_conditions = [c for c in gate.conditions if "par rápido" not in c.label]
    assert all(c.passed for c in other_conditions)


def test_golden_bearish_obv_divergence_fails_the_gate_even_within_an_uptrend():
    # Wyckoff's "effort vs result": price still creeps to marginal new highs,
    # but every third day is a real down-day on heavy volume - the advance
    # is no longer backed by real net buying pressure. Isolates cleanly:
    # every other condition still passes.
    n = 240
    close_base = 100 + np.arange(n) * 0.4
    tail_changes = [-0.3 if i % 3 == 2 else 0.5 for i in range(22)]
    tail_volume = [3_000_000.0 if i % 3 == 2 else 500_000.0 for i in range(22)]
    tail_close = []
    level = close_base[-1]
    for change in tail_changes:
        level += change
        tail_close.append(level)
    close = pd.Series(np.concatenate([close_base, tail_close]))
    volume = pd.Series([1_000_000.0] * n + tail_volume)

    gate = _gate_from_series(close, volume=volume)

    assert _condition(gate, "Tendencia").passed
    assert not _condition(gate, "OBV").passed
    assert not gate.passes
    other_conditions = [c for c in gate.conditions if "OBV" not in c.label]
    assert all(c.passed for c in other_conditions)


def test_golden_parabolic_extension_fails_the_gate_even_with_a_confirmed_uptrend():
    # A genuine, real spike (nearly doubling in the final 5 bars) far above
    # the 50-day average relative to its own ATR - not a normal uptrend
    # continuation. Isolates cleanly: trend/stage still reads a confirmed
    # advance, only the parabolic condition fails.
    base = 100 + np.arange(220) * 0.3
    spike = base[-1] * np.array([1.05, 1.12, 1.22, 1.35, 1.5])
    close = pd.Series(np.concatenate([base, spike]))

    gate = _gate_from_series(close)

    assert _condition(gate, "Tendencia").passed
    assert not _condition(gate, "parabólica").passed
    assert not gate.passes
    other_conditions = [c for c in gate.conditions if "parabólica" not in c.label]
    assert all(c.passed for c in other_conditions)


def test_golden_overbought_spike_within_a_sideways_market_fails_the_gate():
    # RSI pinned extreme (>90) from a sharp bounce, but the broader
    # trend/stage read is still sideways (a longer decline, not yet
    # reclassified as an uptrend by a handful of bars) - "overbought outside
    # a confirmed strong trend" is exactly this case: a short-term thrust
    # inside a market that hasn't actually turned. Not isolated the way the
    # scenarios above are (the parabolic and trend conditions fail too, on
    # this same real chart shape) - a strong-enough short-term thrust to
    # spike RSI this much also spikes ADX and the ATR-multiple in the same
    # 14-bar window, so those three conditions are naturally correlated,
    # not independently triggerable on one fixture.
    decline = 200 - np.arange(220) * 0.3
    bounce = decline[-1] + np.arange(1, 11) * 2.5
    close = pd.Series(np.concatenate([decline, bounce]))

    gate = _gate_from_series(close)

    assert _last(ta.rsi(close)) >= 80
    assert not _condition(gate, "sobrecompra").passed
    assert not gate.passes
