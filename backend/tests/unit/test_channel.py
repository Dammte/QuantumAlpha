"""Parte 4.4 (biblioteca de setups del Radar, quant_methodology.md §28):
canales por regresión lineal. Todos los fixtures verificados con un script
antes de fijarlos (posición real dentro de la banda, no adivinada)."""

from datetime import date

import numpy as np
import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import channel
from app.services.setups.context import SetupContext


def _daily(trend: ta.TrendState = ta.TrendState.UPTREND) -> mtf.TimeframeRead:
    return mtf.TimeframeRead(
        timeframe="daily", trend=trend, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )


def _ctx(close: np.ndarray, weekly: mtf.TimeframeRead | None = None) -> SetupContext:
    close_s = pd.Series(close, dtype=float)
    return SetupContext(
        ticker="XYZ", region="us", trade_date=date(2026, 9, 17),
        close=close_s, high=close_s + 0.5, low=close_s - 0.5, volume=pd.Series([1_000_000.0] * len(close_s)),
        open_=close_s,
        weekly_close=None, weekly_high=None, weekly_low=None, weekly_volume=None,
        atr_series=pd.Series([1.0] * len(close_s)), atr14=1.0, ema21=None, ema55=None,
        sma20=None, sma50=None, sma150=None, sma200=None, rsi14=None,
        levels=[],
        multi_timeframe=mtf.MultiTimeframeRead(
            weekly=weekly, daily=_daily(), intraday=None, alignment="transitioning", alignment_score=0.0,
            conflicts=[],
        ),
        trend=ta.TrendState.UPTREND, weekly_stage=None,
        relative_volume=None, rs_percentile=None, sector_rs_percentile=None, mansfield_rs_series=None,
    )


def test_rising_channel_pullback_to_lower_band():
    rng = np.random.default_rng(7)
    trend = np.arange(60, dtype=float) * 0.5 + 100 + rng.normal(0, 1.0, 60)
    today_near_lower_band = 128.5  # verificado: banda inferior ~128.40, banda superior ~131.89
    close = np.concatenate([trend, [today_near_lower_band]])

    matches = channel.detect(_ctx(close))

    assert len(matches) == 1
    assert matches[0].name == "canal_alcista_banda_inferior"


def test_rising_channel_not_reported_when_price_is_mid_channel():
    rng = np.random.default_rng(7)
    trend = np.arange(60, dtype=float) * 0.5 + 100 + rng.normal(0, 1.0, 60)
    today_mid_channel = 130.1  # a medio camino entre ~128.40 y ~131.89, no en el 20% inferior
    close = np.concatenate([trend, [today_mid_channel]])

    assert channel.detect(_ctx(close)) == []


def test_falling_channel_breakout_when_weekly_bias_is_no_longer_bearish():
    rng = np.random.default_rng(9)
    falling = 150 - np.arange(60, dtype=float) * 0.4 + rng.normal(0, 1.0, 60)
    today_breakout = 129.6  # verificado: banda superior del canal bajista ~127.63
    close = np.concatenate([falling, [today_breakout]])
    weekly_neutral = _daily(trend=ta.TrendState.SIDEWAYS)

    matches = channel.detect(_ctx(close, weekly=weekly_neutral))

    assert len(matches) == 1
    assert matches[0].name == "ruptura_canal_bajista"


def test_falling_channel_breakout_rejected_when_weekly_bias_is_still_bearish():
    rng = np.random.default_rng(9)
    falling = 150 - np.arange(60, dtype=float) * 0.4 + rng.normal(0, 1.0, 60)
    today_breakout = 129.6
    close = np.concatenate([falling, [today_breakout]])
    weekly_bearish = mtf.TimeframeRead(
        timeframe="weekly", trend=ta.TrendState.DOWNTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20="below",
        price_vs_sma50="below", price_vs_sma200="below", bars_since_cross=None,
    )

    assert channel.detect(_ctx(close, weekly=weekly_bearish)) == []


def test_pure_noise_never_produces_a_channel():
    # Ruido i.i.d. sin tendencia genuina - |t| < 2.0, se rechaza antes de
    # llegar a ningún setup concreto (paseo aleatorio de deriva pura NO
    # sirve para este test, ver el mismo comentario en
    # test_technical_analysis.py: por su naturaleza integrada aparenta
    # tendencias que no existen).
    rng = np.random.default_rng(3)
    close = 100 + rng.normal(0, 1.5, 61)

    assert channel.detect(_ctx(close)) == []


def test_detect_returns_empty_without_enough_history():
    close = np.full(30, 100.0)
    assert channel.detect(_ctx(close)) == []
