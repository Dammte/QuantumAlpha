"""Parte 10 (biblioteca de setups del Radar, quant_methodology.md §28):
medición histórica. El mecanismo de reproducción (deduplicación, ventana de
disparo, agregación) se prueba con `setups.registry.detect_all` simulado -
mismo criterio que `test_setups_registry.py`, que también usa detectores
falsos para probar el propio registro sin depender de que un detector real
dispare en una fecha concreta. Los detectores reales ya tienen su propia
suite; este módulo es una capa de orquestación por encima, no un
sustituto de esa cobertura."""

from datetime import UTC, date, datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.domain.models.setup_performance import SetupPerformance
from app.services import setup_replay as sr
from app.services import technical_analysis as ta
from app.services.backtest_engine import TripleBarrierLabel
from app.services.setups import registry as setups_registry
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage


def _ohlcv_df(n_days: int, closes) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    close = pd.Series(closes, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.3, "high": close + 0.6, "low": close - 0.6, "close": close,
            "volume": pd.Series([1_000_000.0] * n_days, index=dates),
        }
    )


def _fake_match(stage: SetupStage, trigger_price: float | None, bars_in_stage: int, name: str = "x") -> SetupMatch:
    return SetupMatch(
        family=SetupFamily.MA_CROSS, name=name, label_es="x", stage=stage, bars_in_stage=bars_in_stage,
        timeframe="daily", trigger_price=trigger_price, trigger_condition="", invalidation_price=None,
        invalidation_condition="", evidence={}, narrative_es="", confidence=SetupConfidence.UNVALIDATED,
    )


def _label(exit_reason: str, return_pct: float, bars_held: int, mae_pct: float = 0.0) -> TripleBarrierLabel:
    return TripleBarrierLabel(
        entry_index=0, exit_index=bars_held, exit_reason=exit_reason, entry_price=100.0,
        exit_price=100.0 * (1 + return_pct), return_pct=return_pct, bars_held=bars_held,
        mae_pct=mae_pct, mfe_pct=abs(return_pct),
    )


def _obs(
    triggered: bool, exit_reason: str | None = None, return_pct: float | None = None,
    bars_held: int | None = None, mae_pct: float = 0.0, grade: str | None = "A",
    regime: str | None = "market_above_sma200", risk_pct: float | None = 0.05,
    setup_name: str = "vcp_3_contracciones", family: str = "vcp",
) -> sr.SetupReplayObservation:
    label = _label(exit_reason, return_pct, bars_held or 1, mae_pct) if triggered and exit_reason else None
    return sr.SetupReplayObservation(
        ticker="X", region="us", setup_name=setup_name, family=family, ready_date=date(2020, 1, 1),
        grade=grade, market_regime=regime, triggered=triggered,
        risk_pct=risk_pct if triggered else None, label=label,
    )


# --- _build_point_in_time_context: sin adelanto de información -------------


def test_point_in_time_context_never_sees_bars_after_i():
    n = 900
    closes = 100 + np.arange(n) * 0.1
    closes[-5:] += 50  # un salto al final, fuera de cualquier ventana temprana
    df = _ohlcv_df(n, closes)
    close, high, low, volume, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    dates = df.index
    atr_series = ta.atr(high, low, close)
    ema21, ema55 = ta.ema(close, 21), ta.ema(close, 55)
    sma20, sma50, sma150, sma200 = ta.sma(close, 20), ta.sma(close, 50), ta.sma(close, 150), ta.sma(close, 200)
    rsi14 = ta.rsi(close)

    ctx = sr._build_point_in_time_context(
        "X", "us", 500, dates, close, high, low, volume, open_, atr_series, ema21, ema55,
        sma20, sma50, sma150, sma200, rsi14, None,
    )

    assert len(ctx.close) == 501
    assert ctx.close.iloc[-1] < 200  # sin el salto final (que empieza en el bar 895), seria ~150
    assert ctx.trade_date == dates[500].date()


def test_point_in_time_context_returns_none_weekly_fields_with_too_little_history():
    n = 5
    df = _ohlcv_df(n, [100.0] * n)
    close, high, low, volume, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    dates = df.index
    atr_series = ta.atr(high, low, close)
    ema21, ema55 = ta.ema(close, 21), ta.ema(close, 55)
    sma20, sma50, sma150, sma200 = ta.sma(close, 20), ta.sma(close, 50), ta.sma(close, 150), ta.sma(close, 200)
    rsi14 = ta.rsi(close)

    # i=0: una sola barra, ninguna semana puede haberse cerrado todavia.
    ctx = sr._build_point_in_time_context(
        "X", "us", 0, dates, close, high, low, volume, open_, atr_series, ema21, ema55,
        sma20, sma50, sma150, sma200, rsi14, None,
    )
    assert ctx.weekly_close is None
    assert ctx.atr14 is None  # ATR14 tampoco esta listo con 1 barra


# --- _find_trigger_bar -------------------------------------------------------


def test_find_trigger_bar_returns_the_first_close_above_the_level():
    close = pd.Series([100.0, 100.5, 99.0, 98.0, 101.5, 102.0])
    assert sr._find_trigger_bar(close, ready_index=0, trigger_price=101.0, window_bars=10) == 4


def test_find_trigger_bar_none_when_never_confirmed_within_the_window():
    close = pd.Series([100.0, 100.5, 99.0, 98.0, 99.5, 100.0])
    assert sr._find_trigger_bar(close, ready_index=0, trigger_price=200.0, window_bars=10) is None


def test_find_trigger_bar_respects_the_window_even_if_confirmed_later():
    close = pd.Series([100.0] * 5 + [200.0])  # confirma en el bar 5, fuera de una ventana de 3
    assert sr._find_trigger_bar(close, ready_index=0, trigger_price=150.0, window_bars=3) is None


# --- replay_setups_for_ticker: mecanismo, con detect_all simulado ----------


def _long_df() -> pd.DataFrame:
    n = 900
    closes = 100 + np.arange(n) * 0.05 + np.random.default_rng(1).normal(0, 0.3, n)
    return _ohlcv_df(n, closes)


def test_replay_records_a_ready_setup_that_goes_on_to_trigger():
    df = _long_df()
    ready_date = df.index[260].date()

    def fake_detect_all(ctx):
        if ctx.trade_date == ready_date:
            return [_fake_match(SetupStage.READY, trigger_price=float(ctx.close.iloc[-1]) + 0.01, bars_in_stage=1)]
        return []

    with patch.object(setups_registry, "detect_all", side_effect=fake_detect_all):
        observations = sr.replay_setups_for_ticker(df, "X", "us", grid_stride_bars=10)

    assert len(observations) == 1
    assert observations[0].triggered is True
    assert observations[0].label is not None


def test_replay_records_a_ready_setup_that_never_triggers():
    df = _long_df()
    ready_date = df.index[260].date()

    def fake_detect_all(ctx):
        if ctx.trade_date == ready_date:
            return [_fake_match(SetupStage.READY, trigger_price=999_999.0, bars_in_stage=1)]
        return []

    with patch.object(setups_registry, "detect_all", side_effect=fake_detect_all):
        observations = sr.replay_setups_for_ticker(df, "X", "us", grid_stride_bars=10)

    assert len(observations) == 1
    assert observations[0].triggered is False
    assert observations[0].label is None


def test_replay_deduplicates_a_ready_setup_that_persists_across_grid_points():
    # Sin la protección de `bars_in_stage`, el mismo READY se contaría una
    # vez por cada punto de la rejilla en que sigue vigente (aquí, ~4 veces
    # con un paso de 10 sesiones sobre una racha de 40).
    df = _long_df()
    ready_start = 260

    def fake_detect_all(ctx):
        bars_since = len(ctx.close) - 1 - ready_start
        if 0 <= bars_since <= 40:
            return [_fake_match(SetupStage.READY, trigger_price=10_000.0, bars_in_stage=bars_since)]
        return []

    with patch.object(setups_registry, "detect_all", side_effect=fake_detect_all):
        observations = sr.replay_setups_for_ticker(df, "X", "us", grid_stride_bars=10)

    assert len(observations) == 1


def test_replay_ignores_forming_and_triggered_stages():
    df = _long_df()
    ready_date = df.index[260].date()

    def fake_detect_all(ctx):
        if ctx.trade_date == ready_date:
            return [
                _fake_match(SetupStage.FORMING, trigger_price=None, bars_in_stage=1, name="forming_x"),
                _fake_match(SetupStage.TRIGGERED, trigger_price=None, bars_in_stage=1, name="triggered_x"),
            ]
        return []

    with patch.object(setups_registry, "detect_all", side_effect=fake_detect_all):
        observations = sr.replay_setups_for_ticker(df, "X", "us", grid_stride_bars=10)

    assert observations == []


def test_replay_returns_empty_with_insufficient_history():
    df = _ohlcv_df(50, [100.0] * 50)
    assert sr.replay_setups_for_ticker(df, "X", "us") == []


# --- aggregate_setup_performance --------------------------------------------


def test_aggregate_computes_the_seven_literal_metrics():
    observations = [
        _obs(True, "target", 0.10, 5, mae_pct=-0.01),
        _obs(True, "target", 0.08, 4, mae_pct=-0.02),
        _obs(True, "stop", -0.05, 2, mae_pct=-0.05),
        _obs(False),
    ]
    stats = sr.aggregate_setup_performance(observations)
    overall = next(s for s in stats if s.grade is None and s.market_regime is None)

    assert overall.n_observations == 4
    assert overall.trigger_rate == pytest.approx(0.75)
    assert overall.win_rate == pytest.approx(2 / 3)
    assert overall.median_bars_held == pytest.approx(4.0)
    assert overall.failure_rate_3d == pytest.approx(1 / 3)  # el stop en 2 barras, dentro de la ventana de 3
    assert overall.expectancy_r is not None
    assert overall.mae_p80_pct is not None and overall.mae_p80_pct < 0


def test_aggregate_confidence_is_measured_at_or_above_the_minimum_sample():
    observations = [_obs(True, "target", 0.05, 5) for _ in range(sr.MIN_SAMPLE_FOR_STATS)]
    stats = sr.aggregate_setup_performance(observations)
    overall = next(s for s in stats if s.grade is None and s.market_regime is None)
    assert overall.confidence == SetupConfidence.MEASURED


def test_aggregate_confidence_is_thin_below_the_minimum_sample():
    observations = [_obs(True, "target", 0.05, 5) for _ in range(sr.MIN_SAMPLE_FOR_STATS - 1)]
    stats = sr.aggregate_setup_performance(observations)
    overall = next(s for s in stats if s.grade is None and s.market_regime is None)
    assert overall.confidence == SetupConfidence.THIN


def test_aggregate_segments_by_grade_and_by_regime_separately_not_jointly():
    observations = [
        _obs(True, "target", 0.05, 5, grade="A", regime="market_above_sma200"),
        _obs(True, "target", 0.05, 5, grade="A", regime="market_above_sma200"),
        _obs(True, "stop", -0.05, 3, grade="B", regime="market_below_sma200"),
    ]
    stats = sr.aggregate_setup_performance(observations)

    keyed = {(s.grade, s.market_regime): s.n_observations for s in stats}
    assert keyed[(None, None)] == 3  # la fila sin segmentar
    assert keyed[("A", None)] == 2
    assert keyed[("B", None)] == 1
    assert keyed[(None, "market_above_sma200")] == 2
    assert keyed[(None, "market_below_sma200")] == 1
    # Nunca una fila que cruce grado Y régimen a la vez - ver el docstring
    # de aggregate_setup_performance.
    assert not any(s.grade is not None and s.market_regime is not None for s in stats)


def test_aggregate_never_fabricates_stats_when_nothing_triggered():
    observations = [_obs(False), _obs(False)]
    stats = sr.aggregate_setup_performance(observations)
    overall = next(s for s in stats if s.grade is None and s.market_regime is None)

    assert overall.n_observations == 2
    assert overall.trigger_rate == 0.0
    assert overall.win_rate is None
    assert overall.expectancy_r is None
    assert overall.median_bars_held is None
    assert overall.mae_p80_pct is None
    assert overall.failure_rate_3d is None


# --- apply_measured_confidence ----------------------------------------------


def _performance_row(setup_name: str, confidence: str, grade: str | None = None) -> SetupPerformance:
    return SetupPerformance(
        id=1, setup_name=setup_name, family="vcp", grade=grade, market_regime=None,
        n_observations=35, trigger_rate=0.6, win_rate=0.55, expectancy_r=0.42,
        median_bars_held=6.0, mae_p80_pct=-0.03, failure_rate_3d=0.1, confidence=confidence,
        computed_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


def test_apply_measured_confidence_upgrades_a_measured_setup():
    match = _fake_match(SetupStage.READY, trigger_price=100.0, bars_in_stage=1, name="vcp_3_contracciones")
    performance_by_name = {"vcp_3_contracciones": _performance_row("vcp_3_contracciones", "measured")}

    result = sr.apply_measured_confidence([match], performance_by_name)

    assert result[0].confidence == SetupConfidence.MEASURED
    assert result[0] is not match  # replace() crea un objeto nuevo, nunca muta el original


def test_apply_measured_confidence_leaves_an_unknown_setup_name_unvalidated():
    match = _fake_match(SetupStage.READY, trigger_price=100.0, bars_in_stage=1, name="nunca_medido")

    result = sr.apply_measured_confidence([match], {"vcp_3_contracciones": _performance_row("x", "measured")})

    assert result[0].confidence == SetupConfidence.UNVALIDATED
    assert result[0] is match  # sin fila que aplicar, se devuelve tal cual


def test_apply_measured_confidence_is_a_noop_with_an_empty_performance_table():
    match = _fake_match(SetupStage.READY, trigger_price=100.0, bars_in_stage=1, name="vcp_3_contracciones")
    assert sr.apply_measured_confidence([match], {}) == [match]


def test_apply_measured_confidence_only_uses_the_ungrouped_row_never_a_segmented_one():
    # `performance_by_name` ya viene filtrado (por daily_close.py) a solo
    # las filas sin segmentar - esta función confía en esa precondición y
    # no vuelve a filtrar por grado, ver su propio docstring.
    match = _fake_match(SetupStage.READY, trigger_price=100.0, bars_in_stage=1, name="vcp_3_contracciones")
    performance_by_name = {"vcp_3_contracciones": _performance_row("vcp_3_contracciones", "thin", grade=None)}

    result = sr.apply_measured_confidence([match], performance_by_name)

    assert result[0].confidence == SetupConfidence.THIN
