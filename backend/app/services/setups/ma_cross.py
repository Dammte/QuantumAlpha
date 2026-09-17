"""Parte 4.3 del encargo: cruce rápido EMA21/EMA55 como setup propio -
distinto del criterio eliminatorio bajista que el gate ya usa
(`levels_engine.evaluate_gate`'s `no_fast_bearish_cross`) y del cruce
proyectado que `exit_engine.py` vigila para posiciones abiertas. Aquí es al
alza, y es una entrada, no una salida.

Reutiliza `multi_timeframe.py` por completo - `ctx.multi_timeframe.daily`
ya trae `cross_quality_20_50` (`technical_analysis.detect_cross_with_quality`,
pese al nombre "20_50" es el par EMA21/55 real, ver ese módulo) e
`imminent_cross_20_50` (`technical_analysis.detect_imminent_cross`) - ambos
calculados una vez por `daily_close.py` para el propio `SetupContext`. Este
detector no recalcula ninguna serie, solo interpreta lo que ya existe."""

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

# Parte 4.3: umbrales propios del setup, deliberadamente distintos de los de
# `detect_cross_with_quality`'s propio criterio de "strong"
# (`CROSS_STRONG_SEPARATION_ATR=0.5`) - ese umbral decide si UN CRUCE YA
# CONFIRMADO amerita confianza alta; este decide si el cruce cuenta como
# setup en absoluto, un listón más bajo a propósito.
MA_CROSS_MIN_SEPARATION_ATR = 0.2
MA_CROSS_MAX_BARS_SINCE = 5  # "en las últimas 5 sesiones" - coincide con CROSS_QUALITY_LOOKBACK
# Más estricto que `IMMINENT_CROSS_MIN_R2=0.5` de la primitiva compartida -
# post-filtro sobre el `r_squared` que esa función ya devuelve, nunca un
# segundo parámetro añadido a `detect_imminent_cross` (que otros
# consumidores siguen usando con su propio 0.5 por defecto).
MA_CROSS_IMMINENT_MIN_R2 = 0.6


def _check_confirmed(cross_quality: ta.CrossQuality | None) -> SetupMatch | None:
    if cross_quality is None or cross_quality.direction != "golden":
        return None
    if cross_quality.bars_since > MA_CROSS_MAX_BARS_SINCE:
        return None
    if cross_quality.separation_atr is None or cross_quality.separation_atr < MA_CROSS_MIN_SEPARATION_ATR:
        return None
    if not (cross_quality.fast_slope > 0 and cross_quality.slow_slope > 0):
        return None

    evidence = {
        "bars_since": cross_quality.bars_since,
        "separation_atr": round(cross_quality.separation_atr, 3),
        "fast_slope_pct": round(cross_quality.fast_slope * 100, 3),
        "slow_slope_pct": round(cross_quality.slow_slope * 100, 3),
    }
    return SetupMatch(
        family=SetupFamily.MA_CROSS,
        name="ma_cross_confirmado",
        label_es="Cruce rápido confirmado (EMA21/55)",
        stage=SetupStage.TRIGGERED,
        bars_in_stage=cross_quality.bars_since,
        timeframe="daily",
        trigger_price=None,
        trigger_condition="EMA21 cruzó por encima de la EMA55, ambas con pendiente positiva",
        invalidation_price=None,
        invalidation_condition="un cruce de vuelta a la baja (EMA21 bajo EMA55) invalida la señal",
        evidence=evidence,
        narrative_es=(
            f"La EMA21 cruzó por encima de la EMA55 hace {cross_quality.bars_since} sesiones, con "
            f"separación de {cross_quality.separation_atr:.2f} ATR y ambas medias subiendo."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def _check_imminent(daily: mtf.TimeframeRead) -> SetupMatch | None:
    imminent = daily.imminent_cross_20_50
    cross_quality = daily.cross_quality_20_50
    if imminent is None or imminent.direction != "golden":
        return None
    if imminent.r_squared < MA_CROSS_IMMINENT_MIN_R2:
        return None
    # La distinción que un detector mecánico suele pasar por alto: un cruce
    # "en 3 sesiones" porque la EMA55 se desploma hacia una EMA21 plana no
    # es la misma señal que una EMA21 subiendo hacia una EMA55 plana o
    # también al alza - solo la segunda es una señal de fuerza real.
    if cross_quality is None or cross_quality.fast_slope <= 0:
        return None

    evidence = {
        "bars_until": imminent.bars_until,
        "r_squared": round(imminent.r_squared, 3),
        "fast_slope_pct": round(cross_quality.fast_slope * 100, 3),
    }
    return SetupMatch(
        family=SetupFamily.MA_CROSS,
        name="ma_cross_proyectado",
        label_es="Cruce rápido proyectado (EMA21/55)",
        stage=SetupStage.READY,
        bars_in_stage=1,
        timeframe="daily",
        trigger_price=None,
        trigger_condition=f"cruce alcista EMA21/55 proyectado en ~{imminent.bars_until} sesiones",
        invalidation_price=None,
        invalidation_condition="la EMA21 se aplana o vuelve a caer antes de completar el cruce",
        evidence=evidence,
        narrative_es=(
            f"Cruce alcista EMA21/55 proyectado en ~{imminent.bars_until} sesiones si continúa la "
            "tendencia actual - la convergencia la produce la EMA21 subiendo, no la EMA55 cayendo."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def detect(ctx: SetupContext) -> list[SetupMatch]:
    daily = ctx.multi_timeframe.daily
    confirmed = _check_confirmed(daily.cross_quality_20_50)
    if confirmed is not None:
        return [confirmed]
    imminent = _check_imminent(daily)
    return [imminent] if imminent is not None else []
