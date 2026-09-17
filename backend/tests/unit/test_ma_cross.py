"""Parte 4.3 (biblioteca de setups del Radar, quant_methodology.md §28):
cruce rápido EMA21/55 al alza como setup propio. Construye `CrossQuality`/
`ImminentCross`/`TimeframeRead` directamente - `ma_cross.py` no recalcula
nada, así que probarlo solo necesita esos objetos ya resueltos, sin series
OHLCV detrás."""

from datetime import date

import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import ma_cross
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupStage


def _cross_quality(
    direction: str = "golden",
    bars_since: int = 2,
    separation_atr: float | None = 0.3,
    fast_slope: float = 0.01,
    slow_slope: float = 0.005,
) -> ta.CrossQuality:
    return ta.CrossQuality(
        direction=direction,
        bars_since=bars_since,
        separation_atr=separation_atr,
        fast_slope=fast_slope,
        slow_slope=slow_slope,
        volume_confirmation=None,
        quality="weak",
    )


def _imminent(direction: str = "golden", bars_until: int = 4, r_squared: float = 0.8) -> ta.ImminentCross:
    return ta.ImminentCross(direction=direction, bars_until=bars_until, r_squared=r_squared)


def _daily(
    cross_quality: ta.CrossQuality | None = None, imminent_cross: ta.ImminentCross | None = None
) -> mtf.TimeframeRead:
    return mtf.TimeframeRead(
        timeframe="daily", trend=ta.TrendState.UPTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=cross_quality, imminent_cross_50_200=None, imminent_cross_20_50=imminent_cross,
        macd_cross=None, macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None,
        price_vs_sma20=None, price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )


def _ctx(daily: mtf.TimeframeRead) -> SetupContext:
    empty = pd.Series(dtype=float)
    return SetupContext(
        ticker="XYZ", region="us", trade_date=date(2026, 9, 17),
        close=empty, high=empty, low=empty, volume=empty, open_=empty,
        weekly_close=None, weekly_high=None, weekly_low=None, weekly_volume=None,
        atr_series=empty, atr14=None, ema21=None, ema55=None, sma20=None, sma50=None, sma150=None, sma200=None,
        levels=[],
        multi_timeframe=mtf.MultiTimeframeRead(
            weekly=None, daily=daily, intraday=None, alignment="transitioning", alignment_score=0.0, conflicts=[]
        ),
        trend=ta.TrendState.UPTREND, weekly_stage=None,
        relative_volume=None, rs_percentile=None, sector_rs_percentile=None, mansfield_rs_series=None,
    )


def test_confirmed_cross_within_5_bars_with_separation_and_both_slopes_up():
    cq = _cross_quality(bars_since=3, separation_atr=0.25, fast_slope=0.02, slow_slope=0.01)
    matches = ma_cross.detect(_ctx(_daily(cross_quality=cq)))

    assert len(matches) == 1
    assert matches[0].name == "ma_cross_confirmado"
    assert matches[0].stage == SetupStage.TRIGGERED
    assert matches[0].bars_in_stage == 3


def test_confirmed_cross_ignored_when_direction_is_death():
    cq = _cross_quality(direction="death", bars_since=1, separation_atr=0.3, fast_slope=-0.01, slow_slope=-0.005)
    assert ma_cross.detect(_ctx(_daily(cross_quality=cq))) == []


def test_confirmed_cross_rejected_when_too_many_bars_have_passed():
    cq = _cross_quality(bars_since=6)  # > MA_CROSS_MAX_BARS_SINCE=5
    assert ma_cross.detect(_ctx(_daily(cross_quality=cq))) == []


def test_confirmed_cross_rejected_when_averages_are_barely_separated():
    cq = _cross_quality(separation_atr=0.1)  # < MA_CROSS_MIN_SEPARATION_ATR=0.2
    assert ma_cross.detect(_ctx(_daily(cross_quality=cq))) == []


def test_confirmed_cross_rejected_when_slow_ma_is_still_falling():
    # Separación de sobra, pero la EMA55 (lenta) todavía cae - "ambas
    # pendientes positivas" no es opcional.
    cq = _cross_quality(fast_slope=0.02, slow_slope=-0.01)
    assert ma_cross.detect(_ctx(_daily(cross_quality=cq))) == []


def test_imminent_cross_when_fast_ma_is_the_one_rising():
    # separación insuficiente para "confirmado" (< 0.2 ATR), así que solo
    # puede coincidir como "proyectado".
    cq_low_sep = _cross_quality(separation_atr=0.05, fast_slope=0.015, slow_slope=0.001)
    imminent = _imminent(bars_until=4, r_squared=0.75)

    matches = ma_cross.detect(_ctx(_daily(cross_quality=cq_low_sep, imminent_cross=imminent)))

    assert len(matches) == 1
    assert matches[0].name == "ma_cross_proyectado"
    assert matches[0].stage == SetupStage.READY


def test_imminent_cross_rejected_when_convergence_comes_from_the_slow_ma_falling():
    """La distinción que Parte 4.3 pide explícitamente: un cruce "en 3
    sesiones" porque la EMA55 se desploma hacia una EMA21 plana no es una
    señal de fuerza - solo cuenta si es la EMA21 (rápida) la que sube."""
    cq = _cross_quality(separation_atr=0.05, fast_slope=-0.001, slow_slope=-0.02)
    imminent = _imminent(bars_until=3, r_squared=0.9)

    assert ma_cross.detect(_ctx(_daily(cross_quality=cq, imminent_cross=imminent))) == []


def test_imminent_cross_rejected_below_the_setups_own_stricter_r2():
    # IMMINENT_CROSS_MIN_R2 de la primitiva compartida es 0.5 - este setup
    # exige 0.6, más estricto, como post-filtro propio (no toca la
    # primitiva compartida, que otros consumidores siguen usando con 0.5).
    cq = _cross_quality(separation_atr=0.05, fast_slope=0.01, slow_slope=0.001)
    imminent = _imminent(bars_until=4, r_squared=0.55)

    assert ma_cross.detect(_ctx(_daily(cross_quality=cq, imminent_cross=imminent))) == []


def test_confirmed_takes_priority_over_imminent_when_both_are_present():
    cq = _cross_quality(bars_since=1, separation_atr=0.3, fast_slope=0.02, slow_slope=0.01)
    imminent = _imminent()  # would also match on its own

    matches = ma_cross.detect(_ctx(_daily(cross_quality=cq, imminent_cross=imminent)))

    assert [m.name for m in matches] == ["ma_cross_confirmado"]


def test_detect_returns_empty_without_any_cross_data():
    assert ma_cross.detect(_ctx(_daily())) == []
