"""Auditoria del Radar, bloque D (docs/quant_methodology.md §29): horizonte
corto/medio plazo. Fixtures verificados con un script de scratchpad antes
de fijarlos - la distancia en ATR y el recorrido medio diario, no
adivinados."""

from datetime import date

import numpy as np
import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import horizon as h
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage


def _daily() -> mtf.TimeframeRead:
    return mtf.TimeframeRead(
        timeframe="daily", trend=ta.TrendState.UPTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )


def _ctx(close: np.ndarray, atr14: float = 2.0) -> SetupContext:
    close_s = pd.Series(close, dtype=float)
    return SetupContext(
        ticker="XYZ", region="us", trade_date=date(2026, 9, 19),
        close=close_s, high=close_s + 0.5, low=close_s - 0.5, volume=pd.Series([1_000_000.0] * len(close_s)),
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


def _match(
    family: SetupFamily = SetupFamily.VCP,
    name: str = "x",
    stage: SetupStage = SetupStage.FORMING,
    timeframe: str = "daily",
    trigger_price: float | None = None,
) -> SetupMatch:
    return SetupMatch(
        family=family, name=name, label_es=name, stage=stage, bars_in_stage=1, timeframe=timeframe,
        trigger_price=trigger_price, trigger_condition="", invalidation_price=None, invalidation_condition="",
        evidence={}, narrative_es="", confidence=SetupConfidence.UNVALIDATED,
    )


# --- assign_horizon: los dos casos literales del encargo (bloque I) ---------


def test_a_forming_vcp_falls_in_medium_horizon():
    # "un setup de VCP en formación cae en medium" - sin gatillo numérico
    # todavía (típico de vcp_forming, que no ha definido pivote).
    close = np.full(60, 100.0)
    ctx = _ctx(close)
    match = _match(family=SetupFamily.VCP, name="vcp_forming", stage=SetupStage.FORMING, trigger_price=None)

    result = h.assign_horizon([match], ctx)

    assert result[0].horizon == "medium"


def test_a_confirmed_breakout_at_0_4_atr_falls_in_short_horizon():
    # "un breakout confirmado a 0.4 ATR cae en short" - literal.
    close = np.full(60, 100.0)
    ctx = _ctx(close, atr14=2.0)
    price = close[-1]
    match = _match(
        family=SetupFamily.BREAKOUT, name="ruptura_de_nivel", stage=SetupStage.READY,
        trigger_price=price + 0.4 * 2.0,  # 0.4 ATR por encima del precio actual
    )

    result = h.assign_horizon([match], ctx)

    assert result[0].horizon == "short"


# --- assign_horizon: resto de las reglas --------------------------------


def test_weekly_timeframe_is_always_medium_even_when_triggered():
    close = np.full(60, 100.0)
    ctx = _ctx(close)
    match = _match(stage=SetupStage.TRIGGERED, timeframe="weekly", trigger_price=100.0)

    result = h.assign_horizon([match], ctx)

    assert result[0].horizon == "medium"


def test_already_triggered_daily_setup_is_always_short_regardless_of_distance():
    close = np.full(60, 100.0)
    ctx = _ctx(close, atr14=2.0)
    # El propio pivote quedó lejos del precio de hoy (una ruptura de hace
    # tiempo), pero "ya disparado hoy" (TRIGGERED) manda igualmente.
    match = _match(stage=SetupStage.TRIGGERED, timeframe="daily", trigger_price=close[-1] - 50.0)

    result = h.assign_horizon([match], ctx)

    assert result[0].horizon == "short"


def test_distance_beyond_one_atr_falls_in_medium():
    close = np.full(60, 100.0)
    ctx = _ctx(close, atr14=2.0)
    match = _match(stage=SetupStage.READY, trigger_price=close[-1] + 5.0)  # 2.5 ATR

    result = h.assign_horizon([match], ctx)

    assert result[0].horizon == "medium"


def test_distance_exactly_at_the_one_atr_threshold_is_short():
    close = np.full(60, 100.0)
    ctx = _ctx(close, atr14=2.0)
    match = _match(stage=SetupStage.READY, trigger_price=close[-1] + 2.0)  # exactamente 1.0 ATR

    result = h.assign_horizon([match], ctx)

    assert result[0].horizon == "short"


def test_missing_atr_never_crashes_and_falls_back_to_medium():
    close = np.full(60, 100.0)
    ctx = _ctx(close, atr14=0.0)  # ATR inválido
    match = _match(stage=SetupStage.READY, trigger_price=close[-1] + 1.0)

    result = h.assign_horizon([match], ctx)

    assert result[0].horizon == "medium"
    assert result[0].expected_sessions_to_trigger is None


# --- expected_sessions_to_trigger --------------------------------------


def test_expected_sessions_is_none_without_a_numeric_trigger():
    close = np.full(60, 100.0)
    ctx = _ctx(close)
    match = _match(trigger_price=None)

    result = h.assign_horizon([match], ctx)

    assert result[0].expected_sessions_to_trigger is None


def test_expected_sessions_is_zero_when_trigger_already_reached():
    close = np.full(60, 100.0)
    ctx = _ctx(close, atr14=2.0)
    match = _match(stage=SetupStage.READY, trigger_price=close[-1])  # distancia 0

    result = h.assign_horizon([match], ctx)

    assert result[0].expected_sessions_to_trigger == 0


def test_expected_sessions_scales_with_distance_not_flat():
    # Dos gatillos a distinta distancia sobre el MISMO histórico de
    # movimiento diario - el más lejano debe estimar más sesiones.
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 60))
    ctx = _ctx(close, atr14=2.0)
    near = _match(name="near", stage=SetupStage.READY, trigger_price=close[-1] + 1.0)
    far = _match(name="far", stage=SetupStage.FORMING, trigger_price=close[-1] + 8.0)

    result = {m.name: m for m in h.assign_horizon([near, far], ctx)}

    assert result["near"].expected_sessions_to_trigger is not None
    assert result["far"].expected_sessions_to_trigger is not None
    assert result["far"].expected_sessions_to_trigger > result["near"].expected_sessions_to_trigger


def test_expected_sessions_is_none_when_price_never_moves():
    # Recorrido medio diario 0 - dividir daría un resultado sin sentido, no
    # infinito fabricado.
    close = np.full(60, 100.0)
    ctx = _ctx(close, atr14=2.0)
    match = _match(stage=SetupStage.READY, trigger_price=close[-1] + 3.0)

    result = h.assign_horizon([match], ctx)

    assert result[0].expected_sessions_to_trigger is None


# --- forma general -------------------------------------------------------


def test_assign_horizon_preserves_every_other_field_unchanged():
    close = np.full(60, 100.0)
    ctx = _ctx(close)
    match = _match(name="vcp_ready", trigger_price=close[-1] + 1.0)

    result = h.assign_horizon([match], ctx)[0]

    assert result.name == match.name
    assert result.family == match.family
    assert result.stage == match.stage
    assert result.confidence == match.confidence


def test_assign_horizon_empty_list_returns_empty_list():
    ctx = _ctx(np.full(60, 100.0))
    assert h.assign_horizon([], ctx) == []
