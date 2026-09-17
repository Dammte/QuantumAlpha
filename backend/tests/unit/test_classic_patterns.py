"""Parte 5 (biblioteca de setups del Radar, quant_methodology.md §28):
patrones clásicos - doble suelo, triángulos, taza con asa y
hombro-cabeza-hombro. Todos los fixtures verificados con un script antes de
fijarlos."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import classic_patterns as cp
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupStage


def _daily() -> mtf.TimeframeRead:
    return mtf.TimeframeRead(
        timeframe="daily", trend=ta.TrendState.UPTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )


def _ctx(close: np.ndarray, volume: np.ndarray, atr14: float = 2.0) -> SetupContext:
    close_s = pd.Series(close, dtype=float)
    return SetupContext(
        ticker="XYZ", region="us", trade_date=date(2026, 9, 17),
        close=close_s, high=close_s + 0.5, low=close_s - 0.5, volume=pd.Series(volume, dtype=float),
        open_=close_s,
        weekly_close=None, weekly_high=None, weekly_low=None, weekly_volume=None,
        atr_series=pd.Series([atr14] * len(close_s)), atr14=atr14, ema21=None, ema55=None,
        sma20=None, sma50=None, sma150=None, sma200=None, rsi14=None,
        levels=[],
        multi_timeframe=mtf.MultiTimeframeRead(
            weekly=None, daily=_daily(), intraday=None, alignment="transitioning", alignment_score=0.0,
            conflicts=[],
        ),
        trend=ta.TrendState.UPTREND, weekly_stage=None,
        relative_volume=None, rs_percentile=None, sector_rs_percentile=None, mansfield_rs_series=None,
    )


def _build(segments: list[tuple[float, float, int]], vol_levels: list[float]) -> tuple[np.ndarray, np.ndarray]:
    close: list[float] = []
    seg_lens: list[int] = []
    for i, (start, end, n) in enumerate(segments):
        seg = np.linspace(start, end, n)
        if i > 0:
            seg = seg[1:]
        close.extend(seg)
        seg_lens.append(len(seg))
    close_arr = np.array(close)
    volume = np.concatenate([np.full(n, v) for n, v in zip(seg_lens, vol_levels, strict=True)])
    return close_arr, volume


# --- Doble suelo ---------------------------------------------------------

_DOUBLE_BOTTOM_SEGMENTS = [(100, 80, 15), (80, 92, 15), (92, 79, 15), (79, 90, 8)]


def test_double_bottom_matches_with_lower_volume_on_the_second_low():
    close, volume = _build(_DOUBLE_BOTTOM_SEGMENTS, [1_000_000.0, 1_000_000.0, 500_000.0, 500_000.0])

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "double_bottom"]

    assert len(matches) == 1
    assert matches[0].stage == SetupStage.READY
    assert matches[0].trigger_price is not None


def test_double_bottom_rejected_when_second_low_has_more_volume():
    close, volume = _build(_DOUBLE_BOTTOM_SEGMENTS, [500_000.0, 500_000.0, 1_500_000.0, 500_000.0])

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "double_bottom"]

    assert matches == []


def test_double_bottom_rejected_when_lows_differ_by_more_than_four_percent():
    # Segundo mínimo un 10% más bajo que el primero - no es "el mismo nivel".
    segments = [(100, 80, 15), (80, 92, 15), (92, 72, 15), (72, 90, 8)]
    close, volume = _build(segments, [1_000_000.0] * 4)

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "double_bottom"]

    assert matches == []


# --- Triángulos ------------------------------------------------------------


def _triangle_close(
    high_level_or_slope: tuple[float, float], low_level_or_slope: tuple[float, float], n: int = 90
) -> np.ndarray:
    """Construye una serie que toca alternadamente una recta "alta" y una
    recta "baja", cada una definida por (valor_inicial, valor_final) sobre
    los primeros 80 de `n` bares - el mismo patrón ya verificado a mano."""
    x = np.arange(n)
    high_start, high_end = high_level_or_slope
    low_start, low_end = low_level_or_slope
    touches_high_idx = [10, 30, 50, 70]
    touches_low_idx = [0, 20, 40, 60, 80]
    pts_idx = sorted(touches_high_idx + touches_low_idx)
    pts_val = []
    for idx in pts_idx:
        frac = idx / 80
        if idx in touches_high_idx:
            pts_val.append(high_start + (high_end - high_start) * frac)
        else:
            pts_val.append(low_start + (low_end - low_start) * frac)
    return np.interp(x, pts_idx, pts_val)


def test_ascending_triangle_flat_top_rising_bottom_can_trigger():
    close = _triangle_close(high_level_or_slope=(150.0, 150.0), low_level_or_slope=(130.0, 148.0))
    volume = np.full(len(close), 1_000_000.0)

    matches = cp.detect(_ctx(close, volume))
    triangle = next((m for m in matches if m.name == "ascending_triangle"), None)

    assert triangle is not None
    assert triangle.trigger_price is not None
    assert triangle.stage == SetupStage.READY


def test_descending_triangle_never_triggers():
    # Techo bajando, suelo plano - Parte 5.6: solo taza-con-asa/doble-suelo/
    # triángulo-ascendente disparan, el resto siempre trigger_price=None.
    close = _triangle_close(high_level_or_slope=(140.0, 105.0), low_level_or_slope=(100.0, 100.0))
    volume = np.full(len(close), 1_000_000.0)

    matches = cp.detect(_ctx(close, volume))
    triangle = next((m for m in matches if m.name == "descending_triangle"), None)

    assert triangle is not None
    assert triangle.trigger_price is None
    assert triangle.invalidation_price is None
    assert triangle.stage == SetupStage.FORMING


def test_pure_noise_produces_no_classic_pattern():
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1.5, 90).cumsum() * 0.1
    volume = np.full(90, 1_000_000.0)

    assert cp.detect(_ctx(close, volume)) == []


def test_detect_returns_empty_with_insufficient_history():
    close = np.full(10, 100.0)
    volume = np.full(10, 1_000_000.0)
    assert cp.detect(_ctx(close, volume)) == []


def test_classic_patterns_can_trigger_whitelist_is_exactly_three_patterns():
    # Bloquea la lista blanca de la Parte 5.6 tal cual - si alguien añade un
    # patrón nuevo a esta familia, este test obliga a decidir explícitamente
    # si dispara o no, en vez de heredarlo por accidente.
    assert cp.CLASSIC_PATTERNS_CAN_TRIGGER == frozenset(
        {"cup_with_handle", "double_bottom", "ascending_triangle"}
    )


# --- Taza con asa ----------------------------------------------------------

_CUP_SEGMENTS = [(100, 75, 30), (75, 76, 26), (76, 98, 35)]
_CUP_VOLUMES = [1_000_000.0, 1_000_000.0, 1_000_000.0]


def _with_handle(
    cup_close: np.ndarray, cup_volume: np.ndarray, handle_end: float
) -> tuple[np.ndarray, np.ndarray]:
    # `_detect_cup_with_handle` mide la forma del asa EXCLUYENDO la barra de
    # "hoy" (ver su propio comentario) - se añade una barra plana extra al
    # nivel de `handle_end` para representar "hoy" sin profundizar ni
    # deshacer el asa ya formada en las barras anteriores, que es lo que
    # este fixture quiere fijar.
    handle_close = np.linspace(cup_close[-1], handle_end, 10)
    handle_close = np.append(handle_close, handle_end)
    handle_volume = np.linspace(1_200_000.0, 700_000.0, 11)
    close = np.concatenate([cup_close, handle_close[1:]])
    volume = np.concatenate([cup_volume, handle_volume[1:]])
    return close, volume


def test_cup_with_handle_matches_with_valid_u_shape_and_declining_handle():
    cup_close, cup_volume = _build(_CUP_SEGMENTS, _CUP_VOLUMES)
    close, volume = _with_handle(cup_close, cup_volume, handle_end=90.0)

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "cup_with_handle"]

    assert len(matches) == 1
    assert matches[0].stage == SetupStage.READY
    assert matches[0].trigger_price == pytest.approx(98.0 * 1.002)
    assert matches[0].evidence["cup_quality"] == "típica"


def test_cup_with_handle_triggers_when_price_closes_above_the_handle_pivot():
    # Ancla de regresión: la primera versión de este detector medía el
    # labio derecho sobre una ventana que incluía la propia barra de hoy -
    # el día de la ruptura, esa barra pasaba a ser "el nuevo labio" y el
    # asa quedaba con duración 0, haciendo el TRIGGERED de este mismo test
    # estructuralmente indetectable. Ver el comentario de `_detect_cup_with_handle`.
    cup_close, cup_volume = _build(_CUP_SEGMENTS, _CUP_VOLUMES)
    close, volume = _with_handle(cup_close, cup_volume, handle_end=90.0)
    close[-1] = 99.5  # por encima de 98 * 1.002

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "cup_with_handle"]

    assert len(matches) == 1
    assert matches[0].stage == SetupStage.TRIGGERED


def test_cup_with_handle_rejected_when_shape_is_a_pure_v():
    # Dos rectas sin ningún tramo plano en el fondo - Parte 5.2: "una V es
    # un fallo, no una base". Ver el comentario de CUP_LOW_ZONE_DEPTH_FRACTION:
    # una V recta pura ya deja ~20% de su ancho en la zona baja por pura
    # geometría, por debajo del 30% exigido.
    cup_close, cup_volume = _build([(100, 75, 40), (75, 98, 40)], [1_000_000.0, 1_000_000.0])
    close, volume = _with_handle(cup_close, cup_volume, handle_end=90.0)

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "cup_with_handle"]

    assert matches == []


def test_cup_with_handle_rejected_when_handle_forms_in_the_lower_half():
    cup_close, cup_volume = _build(_CUP_SEGMENTS, _CUP_VOLUMES)
    # Punto medio de la taza = 75 + 0,5*23 = 86,5 - un asa que baja hasta 82
    # cae en la mitad inferior.
    close, volume = _with_handle(cup_close, cup_volume, handle_end=82.0)

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "cup_with_handle"]

    assert matches == []


def test_cup_with_handle_rejected_when_handle_drifts_up():
    cup_close, cup_volume = _build(_CUP_SEGMENTS, _CUP_VOLUMES)
    close, volume = _with_handle(cup_close, cup_volume, handle_end=104.0)  # "asa" que sube

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "cup_with_handle"]

    assert matches == []


# --- Hombro-cabeza-hombro ---------------------------------------------------


def _head_shoulders_close(
    shoulder_low_or_high_idx: list[int], shoulder_low_or_high_val: list[float],
    neckline_idx: list[int], neckline_val: list[float], n: int = 121,
) -> np.ndarray:
    x = np.arange(n)
    pts_idx = sorted(shoulder_low_or_high_idx + neckline_idx)
    pts_val = []
    for idx in pts_idx:
        if idx in shoulder_low_or_high_idx:
            pts_val.append(shoulder_low_or_high_val[shoulder_low_or_high_idx.index(idx)])
        else:
            pts_val.append(neckline_val[neckline_idx.index(idx)])
    return np.interp(x, pts_idx, pts_val)


def test_head_shoulders_inverse_is_context_only_and_never_sets_a_trigger_price():
    close = _head_shoulders_close(
        [10, 50, 90], [90.0, 80.0, 91.0], [0, 30, 70, 110, 120], [100.0, 100.0, 100.5, 100.0, 102.0],
    )
    volume = np.full(len(close), 1_000_000.0)

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "head_shoulders_inverse"]

    assert len(matches) == 1
    assert matches[0].trigger_price is None
    assert matches[0].stage == SetupStage.TRIGGERED  # cierre (102.0) por encima de la clavicular


def test_head_shoulders_top_is_an_avoid_flag_and_never_sets_a_trigger_price():
    close = _head_shoulders_close(
        [10, 50, 90], [110.0, 120.0, 109.0], [0, 30, 70, 110, 120], [100.0, 100.0, 99.5, 100.0, 99.0],
    )
    volume = np.full(len(close), 1_000_000.0)

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "head_shoulders_top"]

    assert len(matches) == 1
    assert matches[0].trigger_price is None
    assert matches[0].stage == SetupStage.TRIGGERED  # cierre (99.0) por debajo de la clavicular


def test_head_shoulders_forming_when_the_neckline_has_not_broken_yet():
    close = _head_shoulders_close(
        [10, 50, 90], [90.0, 80.0, 91.0], [0, 30, 70, 110, 120], [100.0, 100.0, 100.5, 100.0, 99.0],
    )
    volume = np.full(len(close), 1_000_000.0)

    matches = [m for m in cp.detect(_ctx(close, volume)) if m.name == "head_shoulders_inverse"]

    assert len(matches) == 1
    assert matches[0].stage == SetupStage.FORMING
    assert matches[0].trigger_price is None


def test_head_shoulders_rejected_when_shoulders_are_not_symmetric():
    # Segundo hombro (96) a más de un 5% del primero (90) - Parte 5.5.
    close = _head_shoulders_close(
        [10, 50, 90], [90.0, 80.0, 96.0], [0, 30, 70, 110, 120], [100.0, 100.0, 100.5, 100.0, 102.0],
    )
    volume = np.full(len(close), 1_000_000.0)

    assert [m for m in cp.detect(_ctx(close, volume)) if m.name == "head_shoulders_inverse"] == []
