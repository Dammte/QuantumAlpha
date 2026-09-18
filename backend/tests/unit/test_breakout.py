"""Parte 4.1 (biblioteca de setups del Radar, quant_methodology.md §28):
ruptura de nivel y caja de Darvas."""

from datetime import date

import numpy as np
import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import breakout
from app.services.setups.context import SetupContext


def _level(
    kind: ta.LevelKind = ta.LevelKind.PIVOT_RESISTANCE,
    state: ta.LevelState = ta.LevelState.BROKEN_CONFIRMED,
    side: str = "above",
    strength: int | None = 2,
    distance_atr: float = 0.5,
    bars_in_state: int = 1,
    price: float = 100.0,
) -> ta.Level:
    return ta.Level(
        kind=kind, price=price, side=side, distance_pct=0.02, distance_atr=distance_atr,
        state=state, bars_in_state=bars_in_state, strength=strength, slope_pct_20d=None,
    )


def _ctx(close: np.ndarray, levels: list[ta.Level] | None = None) -> SetupContext:
    close_s = pd.Series(close, dtype=float)
    daily = mtf.TimeframeRead(
        timeframe="daily", trend=ta.TrendState.UPTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )
    return SetupContext(
        ticker="XYZ", region="us", trade_date=date(2026, 9, 17),
        close=close_s, high=close_s + 0.5, low=close_s - 0.5, volume=pd.Series([1_000_000.0] * len(close_s)),
        open_=close_s,
        weekly_close=None, weekly_high=None, weekly_low=None, weekly_volume=None,
        atr_series=pd.Series([1.0] * len(close_s)), atr14=1.0, ema21=None, ema55=None,
        sma20=None, sma50=None, sma150=None, sma200=None, rsi14=None,
        levels=levels or [],
        multi_timeframe=mtf.MultiTimeframeRead(
            weekly=None, daily=daily, intraday=None, alignment="transitioning", alignment_score=0.0, conflicts=[]
        ),
        trend=ta.TrendState.UPTREND, weekly_stage=None,
        relative_volume=None, rs_percentile=None, sector_rs_percentile=None, mansfield_rs_series=None,
    )


def test_level_breakout_matches_a_confirmed_pivot_resistance():
    close = np.full(70, 100.0)
    level = _level()

    matches = breakout.detect(_ctx(close, [level]))

    assert len(matches) == 1
    assert matches[0].name == "ruptura_de_nivel"
    assert matches[0].trigger_price == 100.0


def test_level_breakout_rejected_when_pivot_has_a_single_touch():
    close = np.full(70, 100.0)
    level = _level(strength=1)

    matches = breakout.detect(_ctx(close, [level]))

    assert all(m.name != "ruptura_de_nivel" for m in matches)


def test_range_high_breakout_does_not_need_strength():
    close = np.full(70, 100.0)
    level = _level(kind=ta.LevelKind.RANGE_HIGH_20, strength=None)

    matches = breakout.detect(_ctx(close, [level]))

    assert matches[0].name == "ruptura_de_nivel"


def test_level_breakout_rejected_when_already_too_extended():
    close = np.full(70, 100.0)
    level = _level(distance_atr=1.5)  # > BREAKOUT_MAX_EXTENSION_ATR=1.0

    matches = breakout.detect(_ctx(close, [level]))

    assert all(m.name != "ruptura_de_nivel" for m in matches)


def test_level_breakout_ignores_a_downside_confirmed_level():
    close = np.full(70, 100.0)
    level = _level(side="below")

    matches = breakout.detect(_ctx(close, [level]))

    assert all(m.name != "ruptura_de_nivel" for m in matches)


def test_darvas_box_matches_a_tight_recent_range():
    # 40 sesiones sueltas, luego 30 sesiones apretadas entre 100 y 106 (rango
    # 5,7% < 12%) - la última sesión sigue dentro de la caja.
    loose = 90 + np.arange(40) * 0.3
    tight = 100 + np.sin(np.linspace(0, 6, 30)) * 3  # oscila ~97-103, sin tendencia
    close = np.concatenate([loose, tight])

    matches = breakout.detect(_ctx(close))

    assert len(matches) == 1
    assert matches[0].name == "caja_de_darvas"


def test_darvas_box_9_percent_range_over_30_sessions_triggers_at_the_ceiling_invalidates_at_the_floor():
    # Parte 13.3, escenario 3, literal: "una caja de Darvas del 9% durante
    # 30 sesiones -> gatillo en el techo, anulación en el suelo". 40 sesiones
    # sueltas y anchas (para que las ventanas de 40/50/60 sesiones NO
    # encajen, solo la de 30) seguidas de 30 sesiones apretadas en un rango
    # de ~8,2% (< DARVAS_MAX_RANGE_PCT=0,12, cerca del 9% literal).
    loose = 60 + np.arange(40) * 1.0
    tight = 100 + np.sin(np.linspace(0, 6, 30)) * 4.3
    close = np.concatenate([loose, tight])

    matches = breakout.detect(_ctx(close))

    assert len(matches) == 1
    match = matches[0]
    assert match.name == "caja_de_darvas"
    assert match.evidence["window_sessions"] == 30
    box_high = float(pd.Series(tight).max())
    box_low = float(pd.Series(tight).min())
    assert (box_high - box_low) / box_high < 0.09  # cerca del 9% literal, por debajo
    # "gatillo en el techo, anulación en el suelo" - literal.
    assert match.trigger_price == box_high
    assert match.invalidation_price == box_low


def test_darvas_box_rejected_when_range_is_too_wide():
    loose = 90 + np.arange(40) * 0.3
    wide = 90 + np.arange(30) * 1.0  # rango de 29 sobre un máximo de ~119 -> ~24%, > 12%
    close = np.concatenate([loose, wide])

    assert breakout.detect(_ctx(close)) == []


def test_darvas_box_not_reported_when_todays_close_breaks_above_it():
    # Caja apretada en las 30 sesiones ANTERIORES a la última, y la última
    # sesión cierra claramente por encima del techo de esa caja - la caja ya
    # se rompió, no sigue "vigente".
    loose = 90 + np.arange(39) * 0.3
    tight = 100 + np.sin(np.linspace(0, 6, 29)) * 2  # ~98-102 en las 29 sesiones antes de hoy
    today_breakout = [115.0]
    close = np.concatenate([loose, tight, today_breakout])

    assert breakout.detect(_ctx(close)) == []


def test_level_breakout_takes_priority_over_darvas_box_when_both_apply():
    loose = 90 + np.arange(40) * 0.3
    tight = 100 + np.sin(np.linspace(0, 6, 30)) * 3
    close = np.concatenate([loose, tight])
    level = _level()

    matches = breakout.detect(_ctx(close, [level]))

    assert [m.name for m in matches] == ["ruptura_de_nivel"]
