"""Unit tests for `levels_engine.replay_gate_at` - the point-in-time replay
that replaces the retired `walk_forward_backtest.replay_recommendation_at`
(see docs/quant_methodology.md for the retirement note). Same synthetic
regime-series construction `test_walk_forward_backtest.py` used for its own
`replay_recommendation_at` tests, adapted to the gate's pass/fail shape
instead of a scored verdict."""

import numpy as np
import pandas as pd

from app.services import levels_engine as le
from app.services import technical_analysis as ta


def _synthetic_regime_series(n: int = 800, block: int = 800, seed: int = 123) -> pd.Series:
    """Alternating long uptrend/downtrend regimes, each long enough (relative
    to the 200-day SMA's memory) for `classify_trend` to cleanly read the
    regime it's actually in."""
    rng = np.random.default_rng(seed)
    returns = []
    n_blocks = n // block + 1
    for k in range(n_blocks):
        drift = 0.0025 if k % 2 == 0 else -0.0025
        returns.extend(rng.normal(drift, 0.008, block))
    prices = 100 * np.cumprod(1 + np.array(returns[:n]))
    return pd.Series(prices)


def _indicator_bundle(close: pd.Series) -> dict:
    high = close * 1.005
    low = close * 0.995
    return {
        "close": close,
        "sma20": ta.sma(close, 20),
        "sma50": ta.sma(close, 50),
        "sma150": ta.sma(close, 150),
        "sma200": ta.sma(close, 200),
        "rsi14": ta.rsi(close),
        "adx14": ta.adx(high, low, close),
        "plus_di": ta.dmi(high, low, close)[0],
        "minus_di": ta.dmi(high, low, close)[1],
        "atr14": ta.atr(high, low, close),
    }


def _first_passing_bar(close: pd.Series, bundle: dict, start: int = 250) -> int:
    """Scans for the first bar whose gate genuinely passes, rather than
    hardcoding a specific index - robust to any future threshold tweak in
    `evaluate_gate` that would otherwise silently shift which bars pass."""
    for i in range(start, len(close)):
        result = le.replay_gate_at(i, **bundle)
        if result is not None and result.passes:
            return i
    raise AssertionError("no passing bar found in this synthetic series - fixture needs adjusting")


def test_replay_gate_at_none_before_smas_are_valid():
    close = pd.Series(np.linspace(100, 110, 50))
    bundle = _indicator_bundle(close)
    assert le.replay_gate_at(10, **bundle) is None


def test_replay_gate_at_works_without_volume():
    close = _synthetic_regime_series(n=400)
    bundle = _indicator_bundle(close)
    result = le.replay_gate_at(300, **bundle)
    assert result is not None
    assert isinstance(result.passes, bool)


def test_replay_gate_at_carries_a_stop_loss_when_it_passes():
    # The whole point of replaying the *gate*, not just a verdict string - a
    # passing replay must carry the stop/target evaluate_gate actually
    # proposed at that point.
    close = _synthetic_regime_series()
    bundle = _indicator_bundle(close)
    i = _first_passing_bar(close, bundle)
    result = le.replay_gate_at(i, **bundle)
    assert result.passes is True
    assert result.stop_and_target.stop_loss is not None
    assert result.stop_and_target.stop_loss < close.iloc[i]
    assert result.stop_and_target.take_profit is not None


def test_replay_gate_at_applies_the_fast_pair_veto(monkeypatch):
    # The backtest replay must reflect the same veto the live gate applies,
    # or a backtest would keep simulating a system that no longer exists.
    # Monkeypatches the veto detector itself (already unit-tested on its own
    # in test_technical_analysis.py) rather than hand-constructing a series
    # that happens to satisfy every other condition at once - this test is
    # about the wiring, not re-proving the veto's own detection logic.
    close = _synthetic_regime_series()
    bundle = _indicator_bundle(close)
    i = _first_passing_bar(close, bundle)
    baseline = le.replay_gate_at(i, **bundle)
    assert baseline.passes is True

    monkeypatch.setattr(le, "detect_fast_pair_bearish_veto", lambda close: "señal bajista simulada")
    vetoed = le.replay_gate_at(i, **bundle)
    assert vetoed.passes is False
    veto_condition = next(c for c in vetoed.conditions if "par rápido" in c.label)
    assert veto_condition.passed is False


def test_replay_gate_at_ignores_volume_since_obv_left_the_gate():
    # Sexta auditoría (Parte 6.2, texto literal): OBV nunca fue uno de los 5
    # criterios eliminatorios reales - `volume` se mantiene en la firma por
    # compatibilidad con `find_triple_barrier_entries`'s otros llamadores,
    # pero ya no cambia el resultado del replay en absoluto.
    close = _synthetic_regime_series(n=400)
    bundle = _indicator_bundle(close)
    flat_volume = pd.Series([1000.0] * len(close))
    without_volume = le.replay_gate_at(300, **bundle, volume=None)
    with_flat_volume = le.replay_gate_at(300, **bundle, volume=flat_volume)
    assert without_volume is not None
    assert with_flat_volume is not None
    assert without_volume.passes == with_flat_volume.passes
    assert {c.label for c in without_volume.conditions} == {c.label for c in with_flat_volume.conditions}
