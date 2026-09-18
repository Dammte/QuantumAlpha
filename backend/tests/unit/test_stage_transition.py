"""Parte 2 del encargo (biblioteca de setups del Radar, quant_methodology.md
§28): transición de Weinstein etapa 1 -> 2, los cinco subestados. Todas las
series aquí son SEMANALES a propósito (`weekly_close`/`weekly_volume`) - ver
el propio docstring de `stage_transition.py` para por qué el análisis vive
en esa temporalidad, no en la diaria.

Cada escenario fue verificado numéricamente (pendiente/racha/profundidad
reales) antes de fijar los umbrales de los fixtures - no son solo "parece
razonable", producen los valores intermedios exactos que cada condición
necesita."""

from datetime import date

import numpy as np
import pandas as pd

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups import stage_transition as st
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupStage


def _mtf_stub() -> mtf.MultiTimeframeRead:
    daily = mtf.TimeframeRead(
        timeframe="daily", trend=ta.TrendState.UPTREND, stage=None, ma_cross_50_200=None, ma_cross_20_50=None,
        cross_quality_20_50=None, imminent_cross_50_200=None, imminent_cross_20_50=None, macd_cross=None,
        macd_histogram_turning=None, rsi14=None, adx14=None, di_bias=None, price_vs_sma20=None,
        price_vs_sma50=None, price_vs_sma200=None, bars_since_cross=None,
    )
    return mtf.MultiTimeframeRead(
        weekly=None, daily=daily, intraday=None, alignment="transitioning", alignment_score=0.0, conflicts=[]
    )


def _ctx(
    weekly_close: np.ndarray,
    weekly_volume: np.ndarray,
    daily_close: list[float] | None = None,
    daily_volume: list[float] | None = None,
    atr14: float | None = 1.0,
    relative_volume: float | None = None,
    mansfield_rs: list[float] | None = None,
) -> SetupContext:
    weekly_close_s = pd.Series(weekly_close, dtype=float)
    weekly_volume_s = pd.Series(weekly_volume, dtype=float)
    if daily_close is None:
        daily_close = [float(weekly_close[-1])] * 10
    if daily_volume is None:
        daily_volume = [float(weekly_volume[-1]) / 5] * len(daily_close)
    close = pd.Series(daily_close, dtype=float)
    volume = pd.Series(daily_volume, dtype=float)
    return SetupContext(
        ticker="XYZ",
        region="us",
        trade_date=date(2026, 9, 17),
        close=close,
        high=close + 1,
        low=close - 1,
        volume=volume,
        open_=close,
        weekly_close=weekly_close_s,
        weekly_high=weekly_close_s + 1,
        weekly_low=weekly_close_s - 1,
        weekly_volume=weekly_volume_s,
        atr_series=pd.Series([atr14] * len(close)),
        atr14=atr14,
        ema21=None,
        ema55=None,
        sma20=None,
        sma50=None,
        sma150=None,
        sma200=None,
        rsi14=None,
        levels=[],
        multi_timeframe=_mtf_stub(),
        trend=ta.TrendState.UPTREND,
        weekly_stage=None,
        relative_volume=relative_volume,
        rs_percentile=None,
        sector_rs_percentile=None,
        mansfield_rs_series=pd.Series(mansfield_rs, dtype=float) if mansfield_rs is not None else None,
    )


def _base_weekly_series() -> tuple[np.ndarray, np.ndarray]:
    """40 semanas de caída 200->100, luego 45 semanas planas alrededor de
    100 (con ruido) - una base genuina de etapa 1 ya confirmada
    (`run_weeks=12` sobre las últimas semanas), con el volumen secándose de
    verdad en las últimas semanas. Punto de partida común para los
    escenarios de base confirmada, RS girando y ruptura inminente."""
    rng = np.random.default_rng(3)
    decline = np.linspace(200, 100, 40)
    flat = 100 + rng.normal(0, 0.3, 45)
    weekly_close = np.concatenate([decline, flat])

    decline_vol = np.linspace(3_000_000, 1_500_000, 40)
    flat_vol = np.concatenate([np.linspace(1_500_000, 700_000, 30), np.full(15, 500_000.0)])
    weekly_volume = np.concatenate([decline_vol, flat_vol])
    return weekly_close, weekly_volume


def test_detect_returns_empty_without_enough_weekly_history():
    weekly_close, weekly_volume = _base_weekly_series()
    short_close, short_volume = weekly_close[-20:], weekly_volume[-20:]
    assert st.detect(_ctx(short_close, short_volume)) == []


def test_stage1_base_forming_when_ma30_still_declining_but_flattening():
    # 40 semanas de caída pronunciada, luego 16 semanas de caída mucho más
    # suave (la pendiente de las últimas 8 semanas es menos negativa que la
    # de las 8 anteriores), y un tramo final de 8 semanas apretado (< 25%)
    # con el precio cerca de la MA30.
    decline_steep = np.linspace(200, 110, 40)
    decline_gentle = np.linspace(110, 104, 16)
    tight_tail = np.linspace(104, 102, 8)
    weekly_close = np.concatenate([decline_steep, decline_gentle, tight_tail])
    weekly_volume = np.full(len(weekly_close), 1_000_000.0)

    matches = st.detect(_ctx(weekly_close, weekly_volume))

    assert len(matches) == 1
    assert matches[0].name == "stage1_base_forming"
    assert matches[0].stage == SetupStage.FORMING
    assert matches[0].trigger_price is None  # todavía no hay techo de base definido


def test_stage1_base_forming_does_not_confuse_a_stage3_top_flattening_after_a_rise():
    """El test explícito que la Parte 13.1 exige: una MA30 que se aplana
    tras una SUBIDA (una cima de etapa 3) nunca debe leerse como una base
    de etapa 1 - el discriminador exige que la pendiente de hace 8 semanas
    ya fuera negativa, y una cima de etapa 3 la tiene positiva (el resto de
    la subida)."""
    rise_steep = np.linspace(100, 190, 40)
    rise_gentle = np.linspace(190, 196, 16)  # sigue subiendo, solo más despacio
    tight_tail = np.linspace(196, 197, 8)
    weekly_close = np.concatenate([rise_steep, rise_gentle, tight_tail])
    weekly_volume = np.full(len(weekly_close), 1_000_000.0)

    assert st.detect(_ctx(weekly_close, weekly_volume)) == []


def test_stage1_base_confirmed_with_flat_ma_and_volume_drying_up():
    weekly_close, weekly_volume = _base_weekly_series()

    matches = st.detect(_ctx(weekly_close, weekly_volume))

    assert len(matches) == 1
    assert matches[0].name == "stage1_base_confirmed"
    assert matches[0].stage == SetupStage.FORMING
    assert matches[0].trigger_price is not None
    assert matches[0].invalidation_price is not None
    assert matches[0].invalidation_price < matches[0].trigger_price


def test_stage1_base_confirmed_requires_sustained_volume_dryup_not_a_single_week():
    """Corrección propia (§28.1): una sola semana por debajo del umbral no
    basta - debe sostenerse `STAGE_VOLUME_DRYUP_MIN_WEEKS` semanas seguidas.
    Mismo precio/base que el test anterior, pero el volumen repunta justo
    la última semana en vez de seguir bajo."""
    weekly_close, weekly_volume = _base_weekly_series()
    weekly_volume = weekly_volume.copy()
    weekly_volume[-1] = weekly_volume[-1] * 3  # repunte puntual de una sola semana

    matches = st.detect(_ctx(weekly_close, weekly_volume))

    assert all(m.name != "stage1_base_confirmed" for m in matches)


def test_stage1_rs_turning_when_mansfield_rs_crosses_above_its_own_ma10():
    weekly_close, weekly_volume = _base_weekly_series()
    rng = np.random.default_rng(5)
    # RS oscilando por debajo de su propia MA10 durante 27 sesiones, con un
    # repunte brusco solo en la última - un cruce genuino, no una racha ya
    # en marcha desde antes.
    flat_rs = -3 + rng.normal(0, 0.2, 27)
    mansfield_rs = [*flat_rs.tolist(), 1.0]

    matches = st.detect(_ctx(weekly_close, weekly_volume, mansfield_rs=mansfield_rs))

    assert len(matches) == 1
    assert matches[0].name == "stage1_rs_turning"
    assert matches[0].stage == SetupStage.FORMING


def test_stage1_rs_turning_not_reported_without_a_mansfield_rs_series():
    # Sin benchmark disponible para la región, mansfield_rs_series es None -
    # el subestado simplemente no se evalúa (cae a stage1_base_confirmed),
    # nunca se fabrica.
    weekly_close, weekly_volume = _base_weekly_series()

    matches = st.detect(_ctx(weekly_close, weekly_volume, mansfield_rs=None))

    assert [m.name for m in matches] == ["stage1_base_confirmed"]


def test_stage2_breakout_imminent_within_one_atr_with_rising_volume():
    weekly_close, weekly_volume = _base_weekly_series()
    base_high = float(pd.Series(weekly_close[-12:]).max())
    # Precio a 0,8 ATR del techo (< STAGE_IMMINENT_MAX_DISTANCE_ATR=1.0),
    # todavía por debajo - y volumen relativo alto y subiendo respecto a
    # hace 3 sesiones.
    daily_close = [base_high - 2.5, base_high - 2.0, base_high - 1.5, base_high - 0.8]
    daily_volume = [400_000.0, 420_000.0, 450_000.0, 900_000.0]

    matches = st.detect(
        _ctx(
            weekly_close, weekly_volume, daily_close=daily_close, daily_volume=daily_volume,
            atr14=1.0, relative_volume=1.3,
        )
    )

    assert len(matches) == 1
    assert matches[0].name == "stage2_breakout_imminent"
    assert matches[0].stage == SetupStage.READY


def test_stage2_breakout_imminent_not_reported_when_relative_volume_is_too_low():
    weekly_close, weekly_volume = _base_weekly_series()
    base_high = float(pd.Series(weekly_close[-12:]).max())
    daily_close = [base_high - 2.5, base_high - 2.0, base_high - 1.5, base_high - 0.8]
    daily_volume = [400_000.0, 420_000.0, 450_000.0, 900_000.0]

    matches = st.detect(
        _ctx(
            weekly_close, weekly_volume, daily_close=daily_close, daily_volume=daily_volume,
            atr14=1.0, relative_volume=0.6,  # < 1.0
        )
    )

    assert all(m.name != "stage2_breakout_imminent" for m in matches)


def test_stage2_confirmed_on_a_real_weekly_breakout_with_volume():
    weekly_close, weekly_volume = _base_weekly_series()
    base_high = float(pd.Series(weekly_close[-12:]).max())
    # Ruptura semanal: 6 semanas más, cerrando por encima del techo, con un
    # repunte de volumen fuerte justo en la última.
    breakout_tail = np.linspace(weekly_close[-1], base_high * 1.06, 6)
    weekly_close_breakout = np.concatenate([weekly_close, breakout_tail])
    volume_tail = np.concatenate([np.full(5, 500_000.0), [2_500_000.0]])
    weekly_volume_breakout = np.concatenate([weekly_volume, volume_tail])

    matches = st.detect(_ctx(weekly_close_breakout, weekly_volume_breakout))

    assert len(matches) == 1
    assert matches[0].name == "stage2_confirmed"
    assert matches[0].stage == SetupStage.TRIGGERED
    assert matches[0].timeframe == "weekly"
    assert "semanal" in matches[0].evidence["confirmation"]


def test_stage2_confirmed_daily_only_path_is_marked_as_lower_quality():
    """Parte 2.4: "para operativa diaria, acepta también el cierre diario
    con volumen >= 1,5x la media de 50 días, marcándolo como confirmación
    de menor calidad". Prueba `_check_stage2_confirmed` de forma aislada
    (no a través de `detect()` completo) - construir una serie semanal
    realista donde el techo de la base no esté "contaminado" por la propia
    rampa de ruptura, manteniendo a la vez la pendiente ya positiva, exige
    encajar varias condiciones a la vez sobre una sola serie; probar la
    función directamente con una base y una pendiente ya fijadas aísla
    exactamente la condición que este test quiere verificar."""
    base = st._Base(base_high=100.0, base_low=95.0, depth_pct=0.05, window_weeks=12)
    weekly_close = pd.Series([98.0] * 40 + [99.0])  # nunca cierra por encima del techo semanal
    weekly_volume = pd.Series([500_000.0] * 41)  # sin repunte semanal de volumen
    slope = pd.Series([0.001] * 41)  # MA30 ya con pendiente positiva

    daily_close = [90.0] * 49 + [101.5]  # el cierre diario sí rompe el techo
    daily_volume = [400_000.0] * 49 + [900_000.0]  # >= 1,5x la media de 50 días

    ctx = _ctx(
        weekly_close.to_numpy(), weekly_volume.to_numpy(), daily_close=daily_close, daily_volume=daily_volume,
    )
    match = st._check_stage2_confirmed(ctx, base, slope)

    assert match is not None
    assert match.name == "stage2_confirmed"
    assert match.timeframe == "daily"
    assert "menor calidad" in match.evidence["confirmation"]


def test_stage2_confirmed_requires_ma30_slope_already_positive():
    # Precio ya por encima del techo con volumen de sobra, pero la MA30
    # semanal todavía no ha girado - Parte 2.2 exige las tres condiciones a
    # la vez, "MA30 con pendiente ya positiva" no es opcional.
    base = st._Base(base_high=100.0, base_low=95.0, depth_pct=0.05, window_weeks=12)
    weekly_close = pd.Series([98.0] * 40 + [105.0])
    weekly_volume = pd.Series([500_000.0] * 40 + [5_000_000.0])
    slope = pd.Series([-0.001] * 41)  # todavía negativa

    ctx = _ctx(weekly_close.to_numpy(), weekly_volume.to_numpy())
    assert st._check_stage2_confirmed(ctx, base, slope) is None


def test_the_five_substages_are_reached_in_order_on_a_genuine_decline_to_breakout():
    """Parte 13.3, escenario 1: "un valor que cae, se lateraliza 12 semanas,
    la RS gira, y rompe con volumen -> recorre los cinco subestados en
    orden". Reutiliza exactamente los fixtures ya verificados de cada test
    individual de más arriba (misma caída/base/ruptura, no una serie nueva
    sin probar) para demostrar que, a partir de la MISMA base de 12 semanas
    lateralizada, `detect()` progresa por los cinco nombres en el orden
    correcto según cuánta información adicional se le da."""
    # 1) stage1_base_forming: la MA30 todavía declinando, pero aplanándose -
    # el propio precursor de la base, antes de que exista una base confirmada.
    decline_steep = np.linspace(200, 110, 40)
    decline_gentle = np.linspace(110, 104, 16)
    tight_tail = np.linspace(104, 102, 8)
    forming_close = np.concatenate([decline_steep, decline_gentle, tight_tail])
    forming_volume = np.full(len(forming_close), 1_000_000.0)
    forming = st.detect(_ctx(forming_close, forming_volume))
    assert [m.name for m in forming] == ["stage1_base_forming"]

    # 2) stage1_base_confirmed: la caída se completa y se lateraliza 12
    # semanas de verdad (`_base_weekly_series` - 40 semanas de caída + 45
    # semanas planas con volumen secándose), sin RS todavía.
    weekly_close, weekly_volume = _base_weekly_series()
    confirmed = st.detect(_ctx(weekly_close, weekly_volume))
    assert [m.name for m in confirmed] == ["stage1_base_confirmed"]

    # 3) stage1_rs_turning: la MISMA base, pero la fuerza relativa cruza al
    # alza sobre su propia MA10.
    rng = np.random.default_rng(5)
    flat_rs = -3 + rng.normal(0, 0.2, 27)
    mansfield_rs = [*flat_rs.tolist(), 1.0]
    rs_turning = st.detect(_ctx(weekly_close, weekly_volume, mansfield_rs=mansfield_rs))
    assert [m.name for m in rs_turning] == ["stage1_rs_turning"]

    # 4) stage2_breakout_imminent: la MISMA base, con el precio diario a
    # menos de 1 ATR del techo y volumen relativo alto y subiendo.
    base_high = float(pd.Series(weekly_close[-12:]).max())
    daily_close = [base_high - 2.5, base_high - 2.0, base_high - 1.5, base_high - 0.8]
    daily_volume = [400_000.0, 420_000.0, 450_000.0, 900_000.0]
    imminent = st.detect(
        _ctx(
            weekly_close, weekly_volume, daily_close=daily_close, daily_volume=daily_volume,
            atr14=1.0, relative_volume=1.3,
        )
    )
    assert [m.name for m in imminent] == ["stage2_breakout_imminent"]

    # 5) stage2_confirmed: la MISMA base, ahora con una ruptura semanal real
    # por encima del techo y un repunte de volumen fuerte.
    breakout_tail = np.linspace(weekly_close[-1], base_high * 1.06, 6)
    weekly_close_breakout = np.concatenate([weekly_close, breakout_tail])
    volume_tail = np.concatenate([np.full(5, 500_000.0), [2_500_000.0]])
    weekly_volume_breakout = np.concatenate([weekly_volume, volume_tail])
    confirmed_breakout = st.detect(_ctx(weekly_close_breakout, weekly_volume_breakout))
    assert [m.name for m in confirmed_breakout] == ["stage2_confirmed"]

    # El propio orden narrativo del recorrido, para que este test documente
    # la secuencia y no solo cinco asserts sueltos.
    order = [
        forming[0].name, confirmed[0].name, rs_turning[0].name,
        imminent[0].name, confirmed_breakout[0].name,
    ]
    assert order == [
        "stage1_base_forming", "stage1_base_confirmed", "stage1_rs_turning",
        "stage2_breakout_imminent", "stage2_confirmed",
    ]


def test_a_base_deeper_than_35_percent_is_rejected_as_a_genuine_base():
    # MA30 plana en promedio (zigzag simétrico alrededor de 100), pero el
    # rango real de precios dentro de esa "base" supera el 35% - Parte 2.3:
    # "no es una base, es una tendencia bajista todavía en curso".
    decline = np.linspace(200, 100, 40)
    zigzag = np.array([100 + (30 if i % 2 == 0 else -30) for i in range(45)], dtype=float)
    weekly_close = np.concatenate([decline, zigzag])
    weekly_volume = np.full(len(weekly_close), 1_000_000.0)

    assert st.detect(_ctx(weekly_close, weekly_volume)) == []
