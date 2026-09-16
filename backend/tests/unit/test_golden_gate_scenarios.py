"""Reconstruction (2026-09), Fase 9, reescrito en la Sexta auditoría (texto
literal completo, Parte 6.2): golden-scenario coverage for the gate that
actually decides live now (`levels_engine.evaluate_gate`) - same "hand-built,
textbook-shaped, human-verifiable" philosophy `test_golden_scenarios.py`
established for the retired checklist, reapplied to the 5 literal
eligibility criteria (liquidity_ok/data_quality_ok/weekly_not_stage4/
no_fast_bearish_cross/no_event_risk) that replaced the previous 6-condition
approximation (tendencia/parabólico/sobrecompra/OBV/par rápido/R:R).

Trend/parabolic/overbought/OBV scenarios from the previous version of this
file are gone, not renamed - none of the four are gate *eligibility*
criteria in the literal text (see `levels_engine.py`'s own module
docstring): trend feeds the entry-geometry cascade instead
(`trade_geometry._stop_cascade`), parabolic extension moved to
`exit_engine.py`'s REDUCE trigger for open positions (Parte 9), and OBV
divergence never had literal backing as a gate condition at all."""

import numpy as np
import pandas as pd
import pytest

from app.services import multi_timeframe as mtf
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
    liquidity_ok: bool = True,
    next_earnings_date=None,
    as_of=None,
) -> GateResult:
    """Reproduces the same indicator-derivation pipeline
    `ticker_analysis_service.compute_core_signals` uses before calling the
    real `evaluate_gate` - same role `test_golden_scenarios.py`'s own
    `_recommendation_from_series` played for the retired checklist.
    `close` must carry a real `DatetimeIndex` (business days) - the weekly
    Stage read resamples off it (`multi_timeframe.analyze_multi_timeframe`),
    unlike the retired version of this file, which never needed one."""
    if not isinstance(close.index, pd.DatetimeIndex):
        close = close.set_axis(pd.bdate_range("2015-01-01", periods=len(close)))
    high = high if high is not None else close * 1.01
    low = low if low is not None else close * 0.99
    volume = volume if volume is not None else pd.Series([1_000_000.0] * len(close), index=close.index)

    price = float(close.iloc[-1])
    sma20_s = ta.sma(close, 20)
    trend = ta.classify_trend(price, _last(sma20_s), _last(ta.sma(close, 50)), _last(ta.sma(close, 200)))
    atr14 = _last(ta.atr(high, low, close))
    fast_pair_veto = ta.detect_fast_pair_bearish_veto(close)

    daily_df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})
    multi_timeframe = mtf.analyze_multi_timeframe(daily_df)
    weekly_stage = multi_timeframe.weekly.stage if multi_timeframe.weekly is not None else None

    return evaluate_gate(
        price=price,
        trend=trend,
        atr14=atr14,
        nearest_support=None,
        nearest_resistance=None,
        weekly_stage=weekly_stage,
        liquidity_ok=liquidity_ok,
        fast_pair_bearish_signal=fast_pair_veto,
        next_earnings_date=next_earnings_date,
        as_of=as_of,
    )


def _condition(gate: GateResult, label_substring: str):
    return next(c for c in gate.conditions if label_substring in c.label)


def _clean_uptrend_close(n: int = 700, seed: int = 3) -> pd.Series:
    """Un histórico largo y limpio (>= 60 semanas para un Stage semanal
    confirmado, Parte 5.5) en subida sostenida - el caso base que debería
    aprobar los 5 criterios sin ambigüedad."""
    rng = np.random.default_rng(seed)
    values = 100 + np.arange(n) * 0.3 + rng.normal(0, 1, n).cumsum() * 0.05
    return pd.Series(values, index=pd.bdate_range("2015-01-01", periods=n))


def test_golden_clean_uptrend_passes_every_criterion():
    gate = _gate_from_series(_clean_uptrend_close())
    assert gate.passes
    assert all(c.passed for c in gate.conditions)
    assert gate.eligibility.weekly_not_stage4 is True
    assert gate.stop_and_target.stop_loss < gate.stop_and_target.take_profit


def test_golden_sustained_weekly_decline_fails_on_stage4():
    # Una subida larga seguida de un declive sostenido lo bastante largo como
    # para que la MA30 *semanal* real (no una proxy diaria) ya lea Fase 4 -
    # confirmado contra `multi_timeframe.py` directamente antes de escribir
    # este fixture, no adivinado.
    rng = np.random.default_rng(7)
    n_up = 700
    up = 100 + np.arange(n_up) * 0.3 + rng.normal(0, 1, n_up).cumsum() * 0.05
    n_down = 150
    down = up[-1] - np.arange(1, n_down + 1) * 0.6 + rng.normal(0, 1, n_down).cumsum() * 0.05
    close = pd.Series(np.concatenate([up, down]), index=pd.bdate_range("2015-01-01", periods=n_up + n_down))

    gate = _gate_from_series(close)

    assert gate.eligibility.weekly_not_stage4 is False
    assert not _condition(gate, "Fase 4").passed
    assert not gate.passes


def test_golden_short_history_fails_on_unknown_weekly_stage():
    # Parte 6.2, literal: "unknown NO pasa" - menos de ~60 semanas
    # (`MIN_WEEKLY_BARS_FOR_STAGE`) de historial no puede confirmar ni
    # descartar la Fase 4, así que el criterio falla igual que si estuviera
    # confirmada - nunca se asume "probablemente no está en declive".
    close = pd.Series(100 + np.arange(200) * 0.3, index=pd.bdate_range("2015-01-01", periods=200))
    gate = _gate_from_series(close)
    assert gate.eligibility.weekly_not_stage4 is False
    assert not gate.passes


def test_golden_fast_pair_veto_fails_the_gate_even_with_a_confirmed_uptrend():
    # A long, genuine uptrend that has just begun a smooth, sustained recent
    # decline - still weekly Stage 2 by the slower read, but the fast
    # EMA21/55 pair is already projecting a bearish cross with clean
    # confidence. Isolates cleanly: every other condition still passes.
    up = 100 + np.arange(700) * 0.3
    down = up[-1] - np.arange(1, 41) * 0.2
    close = pd.Series(np.concatenate([up, down]), index=pd.bdate_range("2015-01-01", periods=740))

    gate = _gate_from_series(close)

    assert gate.eligibility.weekly_not_stage4 is True
    assert not _condition(gate, "par rápido").passed
    assert not gate.passes
    other_conditions = [c for c in gate.conditions if "par rápido" not in c.label]
    assert all(c.passed for c in other_conditions)


def test_golden_illiquid_ticker_fails_on_liquidity_even_with_a_clean_uptrend():
    gate = _gate_from_series(_clean_uptrend_close(), liquidity_ok=False)
    assert not _condition(gate, "Liquidez").passed
    assert not gate.passes
    other_conditions = [c for c in gate.conditions if "Liquidez" not in c.label]
    assert all(c.passed for c in other_conditions)


def test_golden_earnings_next_week_fails_on_event_risk():
    as_of = pd.Timestamp("2015-01-01") + pd.tseries.offsets.BDay(699)
    close = _clean_uptrend_close()
    gate = _gate_from_series(
        close, next_earnings_date=as_of.date() + pd.Timedelta(days=5), as_of=as_of.date()
    )
    assert not _condition(gate, "riesgo de evento").passed
    assert not gate.passes
    other_conditions = [c for c in gate.conditions if "riesgo de evento" not in c.label]
    assert all(c.passed for c in other_conditions)


def test_golden_earnings_far_away_still_passes():
    as_of = pd.Timestamp("2015-01-01") + pd.tseries.offsets.BDay(699)
    close = _clean_uptrend_close()
    gate = _gate_from_series(
        close, next_earnings_date=as_of.date() + pd.Timedelta(days=60), as_of=as_of.date()
    )
    assert gate.eligibility.no_event_risk is True
    assert gate.passes


def test_golden_downtrend_no_longer_fails_the_gate_by_itself():
    # Parte 6.2 (literal): la dirección de la tendencia diaria ya no es un
    # criterio de elegibilidad - solo el estado *semanal* (Fase 4) lo es.
    # Un declive diario reciente, corto, dentro de un historial semanal
    # todavía no confirmado en Fase 4, no descalifica por sí solo (aunque en
    # la práctica raramente producirá un disparador viable, ya que la
    # cascada de la geometría exige tendencia alcista para sus peldaños de
    # continuación - ver trade_geometry._stop_cascade).
    close = pd.Series(200 - np.arange(300) * 0.05, index=pd.bdate_range("2015-01-01", periods=300))
    gate = _gate_from_series(close)
    # Ni "Tendencia" ni "parabólica" ni "sobrecompra" existen ya como
    # etiquetas del gate - solo los 5 criterios literales.
    labels = {c.label for c in gate.conditions}
    assert len(labels) == 5
    assert not any("Tendencia" in label or "parabólica" in label or "sobrecompra" in label for label in labels)


@pytest.mark.parametrize("liquidity_ok", [True, False])
def test_golden_eligibility_object_matches_the_conditions_list(liquidity_ok):
    gate = _gate_from_series(_clean_uptrend_close(), liquidity_ok=liquidity_ok)
    assert gate.eligibility.liquidity_ok is liquidity_ok
    assert gate.passes == gate.eligibility.passes == all(c.passed for c in gate.conditions)
