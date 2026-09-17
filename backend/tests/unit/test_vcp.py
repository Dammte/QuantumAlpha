"""Parte 3 (biblioteca de setups del Radar, quant_methodology.md §28): VCP.
Todos los fixtures verificados con un script antes de fijarlos - la
secuencia de contracciones, el volumen y el disparo/fallo, no adivinados."""

from datetime import date

import numpy as np
import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import vcp
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
    """Construye una serie de cierres por tramos lineales, evitando
    duplicar el punto de unión entre tramos consecutivos (que rompería el
    detector de pivotes al crear un "empate" justo en el pico/valle)."""
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


# Tres contracciones limpias, decrecientes: 24% -> 13,03% -> 7,03% (el
# ejemplo canónico de Minervini es 25%/15%/8% - esta serie está deliberada-
# mente cerca de eso). Volumen decreciente en cada tramo sucesivo.
_THREE_CONTRACTIONS_SEGMENTS = [
    (100, 130, 15), (130, 98.8, 15), (98.8, 122, 15), (122, 106.1, 15),
    (106.1, 118, 15), (118, 109.7, 15),
]
_THREE_CONTRACTIONS_VOLUME = [3_000_000.0, 2_500_000.0, 1_800_000.0, 1_200_000.0, 900_000.0, 500_000.0]


def _three_contractions_with_tail(tail_price: float, tail_bars: int, tail_volume: float):
    """Las tres contracciones ya verificadas, más un tramo final que
    aproxima o rompe el pivote (118,0), según el escenario de cada test."""
    return _build(
        [*_THREE_CONTRACTIONS_SEGMENTS, (109.7, tail_price, tail_bars)],
        [*_THREE_CONTRACTIONS_VOLUME, tail_volume],
    )


def test_vcp_ready_with_three_decreasing_contractions_near_the_pivot():
    close, volume = _three_contractions_with_tail(116.0, 8, 600_000.0)

    matches = vcp.detect(_ctx(close, volume))

    assert len(matches) == 1
    assert matches[0].name == "vcp_ready"
    assert matches[0].stage == SetupStage.READY
    assert matches[0].evidence["n_contractions"] == 3
    assert matches[0].evidence["depths_pct"] == [0.24, 0.1303, 0.0703]


def test_vcp_forming_with_only_two_contractions():
    # [:5], no [:4]: el segundo mínimo necesita un tramo posterior que
    # vuelva a subir para poder confirmarse como pivote real (un mínimo en
    # el último bar de la serie, sin nada después, nunca se confirma).
    close, volume = _build(_THREE_CONTRACTIONS_SEGMENTS[:5], _THREE_CONTRACTIONS_VOLUME[:5])

    matches = vcp.detect(_ctx(close, volume))

    assert len(matches) == 1
    assert matches[0].name == "vcp_forming"
    assert matches[0].stage == SetupStage.FORMING


def test_vcp_triggered_on_a_close_above_the_pivot_with_strong_volume():
    close, volume = _three_contractions_with_tail(122.0, 8, 550_000.0)
    volume = volume.copy()
    volume[-1] = 3_000_000.0  # repunte de volumen claro en la ruptura

    matches = vcp.detect(_ctx(close, volume))

    assert len(matches) == 1
    assert matches[0].name == "vcp_triggered"
    assert matches[0].stage == SetupStage.TRIGGERED


def test_vcp_failed_when_price_falls_back_below_the_pivot():
    close, volume = _three_contractions_with_tail(122.0, 8, 550_000.0)
    close = close.copy()
    # Disparó (llegó a 122, por encima del pivote 118) y hoy cae de vuelta
    # por debajo del propio pivote.
    close[-1] = 115.0

    matches = vcp.detect(_ctx(close, volume))

    assert len(matches) == 1
    assert matches[0].name == "vcp_failed"
    assert matches[0].stage == SetupStage.FAILED
    assert matches[0].trigger_price is None  # ya falló, no hay nada que disparar


def test_vcp_rejected_when_contractions_are_not_decreasing():
    # Tercera contracción (20%) más ancha que la segunda (13%) - crece en
    # vez de contraerse, no es un VCP.
    segments = [
        (100, 130, 15), (130, 98.8, 15), (98.8, 122, 15), (122, 106.1, 15),
        (106.1, 118, 15), (118, 94.4, 15), (94.4, 100.0, 8),
    ]
    close, volume = _build(segments, [3_000_000.0] * 7)

    assert vcp.detect(_ctx(close, volume)) == []


def test_vcp_rejected_when_volume_increases_instead_of_decreasing():
    volume_increasing = [1_000_000.0, 500_000.0, 1_000_000.0, 900_000.0, 1_000_000.0, 1_500_000.0, 1_000_000.0]
    close, volume = _build([*_THREE_CONTRACTIONS_SEGMENTS, (109.7, 116.0, 8)], volume_increasing)

    assert vcp.detect(_ctx(close, volume)) == []


def test_vcp_rejected_with_insufficient_history():
    close = np.full(20, 100.0)
    volume = np.full(20, 1_000_000.0)
    assert vcp.detect(_ctx(close, volume)) == []


def test_vcp_rejected_with_too_few_bars_for_any_contraction():
    close = np.linspace(100, 110, 10)
    volume = np.full(10, 1_000_000.0)
    assert vcp.detect(_ctx(close, volume)) == []
