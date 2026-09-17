"""Parte 4.2 (biblioteca de setups del Radar, quant_methodology.md §28):
retroceso en tendencia. Construye `Level` directamente (el motor de niveles
ya tiene sus propios 20 tests en `test_technical_analysis.py` - aquí solo
importa que `pullback.py` interprete el estado correctamente, no
recalcularlo)."""

from datetime import date

import numpy as np
import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import pullback
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupStage


def _level(
    kind: ta.LevelKind = ta.LevelKind.EMA21,
    state: ta.LevelState = ta.LevelState.TESTING,
    side: str = "above",
    strength: int | None = None,
    distance_atr: float = 0.3,
    bars_in_state: int = 2,
) -> ta.Level:
    return ta.Level(
        kind=kind, price=100.0, side=side, distance_pct=0.01, distance_atr=distance_atr,
        state=state, bars_in_state=bars_in_state, strength=strength, slope_pct_20d=None,
    )


def _ctx(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    levels: list[ta.Level],
    trend: ta.TrendState = ta.TrendState.UPTREND,
    weekly: mtf.TimeframeRead | None = None,
    rsi14: float | None = 45.0,
    atr14: float | None = 1.0,
) -> SetupContext:
    close_s = pd.Series(close, dtype=float)
    daily = mtf.TimeframeRead(
        timeframe="daily", trend=trend, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )
    return SetupContext(
        ticker="XYZ", region="us", trade_date=date(2026, 9, 17),
        close=close_s, high=pd.Series(high, dtype=float), low=pd.Series(low, dtype=float),
        volume=pd.Series(volume, dtype=float), open_=close_s,
        weekly_close=None, weekly_high=None, weekly_low=None, weekly_volume=None,
        atr_series=pd.Series([atr14] * len(close_s)), atr14=atr14, ema21=None, ema55=None,
        sma20=None, sma50=None, sma150=None, sma200=None, rsi14=rsi14,
        levels=levels,
        multi_timeframe=mtf.MultiTimeframeRead(
            weekly=weekly, daily=daily, intraday=None, alignment="transitioning", alignment_score=0.0, conflicts=[]
        ),
        trend=trend, weekly_stage=None,
        relative_volume=None, rs_percentile=None, sector_rs_percentile=None, mansfield_rs_series=None,
    )


def _impulse_and_pullback_prices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Impulso de 90 a 110 en 30 sesiones, luego un retroceso a 104 (30% del
    impulso de 20 puntos - dentro del 50% permitido)."""
    impulse = np.linspace(90, 110, 30)
    pull = np.linspace(110, 104, 5)
    close = np.concatenate([impulse, pull])
    high = close + 0.5
    low = close - 0.5
    return close, high, low


def _drying_volume(n: int) -> np.ndarray:
    # Últimos 3 días muy por debajo de la media de 20 - ratio bien < 0.75.
    return np.concatenate([np.full(n - 3, 1_000_000.0), np.full(3, 400_000.0)])


def test_pullback_matches_when_all_conditions_hold():
    close, high, low = _impulse_and_pullback_prices()
    volume = _drying_volume(len(close))
    level = _level()

    matches = pullback.detect(_ctx(close, high, low, volume, [level]))

    assert len(matches) == 1
    assert matches[0].name == "pullback_a_media_o_soporte"
    assert matches[0].stage == SetupStage.READY
    assert matches[0].invalidation_price < matches[0].trigger_price


def test_pullback_not_reported_outside_an_uptrend():
    close, high, low = _impulse_and_pullback_prices()
    volume = _drying_volume(len(close))
    level = _level()

    matches = pullback.detect(_ctx(close, high, low, volume, [level], trend=ta.TrendState.SIDEWAYS))

    assert matches == []


def test_pullback_not_reported_when_weekly_bias_is_bearish():
    close, high, low = _impulse_and_pullback_prices()
    volume = _drying_volume(len(close))
    level = _level()
    bearish_weekly = mtf.TimeframeRead(
        timeframe="weekly", trend=ta.TrendState.DOWNTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20="below",
        price_vs_sma50="below", price_vs_sma200="below", bars_since_cross=None,
    )

    matches = pullback.detect(_ctx(close, high, low, volume, [level], weekly=bearish_weekly))

    assert matches == []


def test_pullback_not_reported_without_a_matching_level():
    close, high, low = _impulse_and_pullback_prices()
    volume = _drying_volume(len(close))
    far_level = _level(state=ta.LevelState.FAR)

    assert pullback.detect(_ctx(close, high, low, volume, [far_level])) == []


def test_pullback_requires_pivot_support_to_have_at_least_two_touches():
    close, high, low = _impulse_and_pullback_prices()
    volume = _drying_volume(len(close))
    weak_pivot = _level(kind=ta.LevelKind.PIVOT_SUPPORT, strength=1)

    assert pullback.detect(_ctx(close, high, low, volume, [weak_pivot])) == []

    strong_pivot = _level(kind=ta.LevelKind.PIVOT_SUPPORT, strength=2)
    assert pullback.detect(_ctx(close, high, low, volume, [strong_pivot]))[0].name == "pullback_a_media_o_soporte"


def test_pullback_not_reported_when_volume_is_not_drying_up():
    close, high, low = _impulse_and_pullback_prices()
    volume = np.full(len(close), 1_000_000.0)  # sin secado
    level = _level()

    assert pullback.detect(_ctx(close, high, low, volume, [level])) == []


def test_pullback_not_reported_when_rsi_is_outside_the_40_55_band():
    close, high, low = _impulse_and_pullback_prices()
    volume = _drying_volume(len(close))
    level = _level()

    assert pullback.detect(_ctx(close, high, low, volume, [level], rsi14=25.0)) == []
    assert pullback.detect(_ctx(close, high, low, volume, [level], rsi14=70.0)) == []


def test_pullback_not_reported_when_retracement_exceeds_50_percent():
    impulse = np.linspace(90, 110, 30)
    deep_pull = np.linspace(110, 99, 5)  # retrocede 11 de 20 = 55% > 50%
    close = np.concatenate([impulse, deep_pull])
    high = close + 0.5
    low = close - 0.5
    volume = _drying_volume(len(close))
    level = _level()

    assert pullback.detect(_ctx(close, high, low, volume, [level])) == []
