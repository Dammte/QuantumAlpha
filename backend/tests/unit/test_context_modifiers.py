"""Parte 6 (biblioteca de setups del Radar, quant_methodology.md §28):
modificadores de contexto. Todos los fixtures verificados con un script
antes de fijarlos - en particular el margen ATR de la ruptura fallida, que
no existía en la primera versión y producía falsos positivos en ruido
lateral puro (ver el comentario de `_detect_recent_failed_breakout`)."""

from datetime import date

import numpy as np
import pandas as pd

from app.services import levels_engine as le
from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import context_modifiers as cm
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage


def _daily() -> mtf.TimeframeRead:
    return mtf.TimeframeRead(
        timeframe="daily", trend=ta.TrendState.UPTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )


def _ctx(
    close: np.ndarray, volume: np.ndarray, atr14: float = 2.0, atr_series: np.ndarray | None = None
) -> SetupContext:
    close_s = pd.Series(close, dtype=float)
    atr_s = pd.Series(atr_series, dtype=float) if atr_series is not None else pd.Series([atr14] * len(close_s))
    return SetupContext(
        ticker="XYZ", region="us", trade_date=date(2026, 9, 17),
        close=close_s, high=close_s + 0.5, low=close_s - 0.5, volume=pd.Series(volume, dtype=float),
        open_=close_s,
        weekly_close=None, weekly_high=None, weekly_low=None, weekly_volume=None,
        atr_series=atr_s, atr14=atr14, ema21=None, ema55=None,
        sma20=None, sma50=None, sma150=None, sma200=None, rsi14=None,
        levels=[],
        multi_timeframe=mtf.MultiTimeframeRead(
            weekly=None, daily=_daily(), intraday=None, alignment="transitioning", alignment_score=0.0,
            conflicts=[],
        ),
        trend=ta.TrendState.UPTREND, weekly_stage=None,
        relative_volume=None, rs_percentile=None, sector_rs_percentile=None, mansfield_rs_series=None,
    )


def _match(stage: SetupStage) -> SetupMatch:
    return SetupMatch(
        family=SetupFamily.STAGE_TRANSITION, name="x", label_es="x", stage=stage, bars_in_stage=1,
        timeframe="daily", trigger_price=None, trigger_condition="", invalidation_price=None,
        invalidation_condition="", evidence={}, narrative_es="", confidence=SetupConfidence.UNVALIDATED,
    )


# --- Squeeze de volatilidad -------------------------------------------------


def test_detect_squeeze_true_when_bollinger_sits_inside_keltner():
    rng = np.random.default_rng(1)
    tight = 100 + rng.normal(0, 0.05, 80)
    assert cm._detect_squeeze(_ctx(tight, np.full(80, 1_000_000.0))) is True


def test_detect_squeeze_false_in_a_trending_series():
    # Una tendencia limpia ensancha Bollinger (la desviación estándar de los
    # cierres crece con el propio recorrido de la tendencia) mucho más de lo
    # que ensancha el ATR (que solo ve el paso día a día) - el caso opuesto
    # al squeeze, sin depender de una amplitud de ruido concreta.
    trending = np.linspace(100, 150, 80)
    assert cm._detect_squeeze(_ctx(trending, np.full(80, 1_000_000.0))) is False


# --- Pocket pivot ------------------------------------------------------------

# 1 barra de apoyo (para el primer diff) + 10 sesiones previas alternando
# (la mayor bajista con volumen 900k) + una última sesión que varía por test.
_POCKET_BASE = [98.0]
_POCKET_PRIOR_CLOSE = [100, 99, 100.5, 99.5, 101, 100, 101.5, 100.5, 102, 101]
_POCKET_PRIOR_VOLUME = [500e3, 900e3, 400e3, 700e3, 300e3, 600e3, 350e3, 650e3, 300e3, 500e3]


def test_detect_pocket_pivot_true_with_up_day_high_volume_and_price_above_sma10():
    close = _POCKET_BASE + _POCKET_PRIOR_CLOSE + [103.0]
    volume = [500e3] + _POCKET_PRIOR_VOLUME + [1_500_000.0]
    assert cm._detect_pocket_pivot(_ctx(close, volume)) is True


def test_detect_pocket_pivot_false_when_volume_does_not_exceed_the_prior_down_day():
    close = _POCKET_BASE + _POCKET_PRIOR_CLOSE + [103.0]
    volume = [500e3] + _POCKET_PRIOR_VOLUME + [300_000.0]  # < 900k, el mayor volumen bajista previo
    assert cm._detect_pocket_pivot(_ctx(close, volume)) is False


def test_detect_pocket_pivot_false_on_a_down_day():
    close = _POCKET_BASE + _POCKET_PRIOR_CLOSE + [95.0]
    volume = [500e3] + _POCKET_PRIOR_VOLUME + [1_500_000.0]
    assert cm._detect_pocket_pivot(_ctx(close, volume)) is False


# --- Secado de volumen -------------------------------------------------------


def test_detect_volume_dryup_true_when_recent_average_drops_below_seventy_percent():
    volume = np.concatenate([np.full(45, 1_000_000.0), np.full(5, 500_000.0)])
    assert cm._detect_volume_dryup(_ctx(np.full(50, 100.0), volume)) is True


def test_detect_volume_dryup_false_with_constant_volume():
    assert cm._detect_volume_dryup(_ctx(np.full(50, 100.0), np.full(50, 1_000_000.0))) is False


# --- Contracción de ATR -------------------------------------------------------


def test_detect_atr_contraction_true_when_atr14_is_well_below_its_own_average():
    atr_series = np.concatenate([np.full(49, 2.0), [1.0]])
    ctx = _ctx(np.full(50, 100.0), np.full(50, 1_000_000.0), atr14=1.0, atr_series=atr_series)
    assert cm._detect_atr_contraction(ctx) is True


def test_detect_atr_contraction_false_without_contraction():
    ctx = _ctx(np.full(50, 100.0), np.full(50, 1_000_000.0), atr14=2.0, atr_series=np.full(50, 2.0))
    assert cm._detect_atr_contraction(ctx) is False


# --- Ruptura fallida reciente -------------------------------------------------

_FB_SEG0 = [94.0] * 10
_FB_SEG1 = [95, 96, 97, 98, 99, 100, 99, 98, 97, 96]  # pivote de fractal genuino en 100
_FB_SEG2 = [96.0] * 5


def test_detect_recent_failed_breakout_true_after_a_quick_reversal():
    close = _FB_SEG0 + _FB_SEG1 + _FB_SEG2 + [101.0] + [99.0, 95.0] + [95.0] * 5
    assert cm._detect_recent_failed_breakout(_ctx(close, np.full(len(close), 1_000_000.0))) is True


def test_detect_recent_failed_breakout_false_when_the_breakout_holds():
    close = _FB_SEG0 + _FB_SEG1 + _FB_SEG2 + [101.0] + [102.0, 103.0] + [104.0] * 5
    assert cm._detect_recent_failed_breakout(_ctx(close, np.full(len(close), 1_000_000.0))) is False


def test_detect_recent_failed_breakout_false_on_pure_noise_without_real_structure():
    # Ancla de regresión: la primera versión (sin margen ATR) marcaba esto
    # en True - cualquier pivote diminuto de una serie lateral ruidosa se
    # "rompía" y se "perdía" por pura varianza, sin ninguna ruptura real.
    rng = np.random.default_rng(3)
    flat_noisy = 100 + rng.normal(0, 0.05, 80)
    assert cm._detect_recent_failed_breakout(_ctx(flat_noisy, np.full(80, 1_000_000.0))) is False


# --- Coincidencia de setups ---------------------------------------------------


def test_detect_setup_coincidence_counts_only_ready_not_triggered():
    setups = [_match(SetupStage.READY), _match(SetupStage.READY), _match(SetupStage.TRIGGERED)]
    assert cm._detect_setup_coincidence(setups) == 2  # TRIGGERED no cuenta, literal del encargo


# --- apply_context_modifiers ---------------------------------------------------


def test_apply_context_modifiers_caps_multiple_upgrades_to_a_single_step():
    # Rampa suave y estrictamente creciente: sin pivotes de alto interiores,
    # así que no interfiere con la ruptura fallida - solo fuerza squeeze +
    # contracción de ATR, junto con 3 setups en READY (coincidencia).
    smooth_ramp = 100 + np.linspace(0, 0.5, 80)
    atr_series = np.concatenate([np.full(79, 2.0), [1.0]])
    ctx = _ctx(smooth_ramp, np.full(80, 1_000_000.0), atr14=1.0, atr_series=atr_series)
    setups = [_match(SetupStage.READY)] * 3

    result = cm.apply_context_modifiers(le.GradeResult(grade=le.Grade.B, reasons=["base B"]), ctx, setups)

    assert result.grade == le.Grade.A  # B + 1 escalón, nunca más aunque 3 razones se cumplan a la vez
    assert len(result.reasons) == 4  # la razón base + exactamente 3 (squeeze, ATR, coincidencia)


def test_apply_context_modifiers_recent_failed_breakout_caps_to_c_even_from_a():
    close = _FB_SEG0 + _FB_SEG1 + _FB_SEG2 + [101.0] + [99.0, 95.0] + [95.0] * 5
    ctx = _ctx(close, np.full(len(close), 1_000_000.0))

    result = cm.apply_context_modifiers(le.GradeResult(grade=le.Grade.A, reasons=["base A"]), ctx, [])

    assert result.grade == le.Grade.C


def test_apply_context_modifiers_is_a_noop_when_grade_is_none():
    grade_result = le.GradeResult(grade=None, reasons=["geometría no viable"])
    close = _FB_SEG0 + _FB_SEG1 + _FB_SEG2 + [101.0] + [99.0, 95.0] + [95.0] * 5
    ctx = _ctx(close, np.full(len(close), 1_000_000.0))

    result = cm.apply_context_modifiers(grade_result, ctx, [])

    assert result.grade is None
    assert result.reasons == ["geometría no viable"]


def test_apply_context_modifiers_pocket_pivot_and_volume_dryup_never_change_the_grade():
    # El encargo solo dice "marca .../confirma ..." para estos dos, sin
    # ningún "puede subir un escalón" (a diferencia de squeeze/ATR/
    # coincidencia) - deben aparecer como razón, pero nunca mover el grado.
    close = _POCKET_BASE + _POCKET_PRIOR_CLOSE + [103.0]
    volume = [500e3] + _POCKET_PRIOR_VOLUME + [1_500_000.0]
    ctx = _ctx(close, volume)
    assert cm._detect_pocket_pivot(ctx) is True

    result = cm.apply_context_modifiers(le.GradeResult(grade=le.Grade.B, reasons=["base B"]), ctx, [])

    assert result.grade == le.Grade.B
    assert any("Pocket pivot" in r for r in result.reasons)
