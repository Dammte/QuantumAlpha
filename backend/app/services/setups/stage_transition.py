"""Parte 2 del encargo: transición de Weinstein etapa 1 -> 2 - "la petición
central del propietario", y el setup con mejor relación entre lo que aporta
y lo que cuesta calcular, porque captura el movimiento antes de que sea
obvio. Ver `setups/__init__.py` para el estándar de evidencia de toda la
biblioteca.

Todo el análisis va sobre barras SEMANALES - la MA30 semanal
(`multi_timeframe.WEEKLY_STAGE_MA_WINDOW`), no una proxy diaria (esa
distinción, y por qué importa, ya está resuelta en el propio gate desde la
Sexta auditoría, ver `docs/quant_methodology.md` §27.6).

Atribución: las cuatro etapas son de Stan Weinstein; Minervini las usa
explícitamente en su propio marco y añade el VCP (`setups/vcp.py`, fase
posterior) como el patrón concreto que se forma al final de la etapa 1 -
Weinstein da el contexto, VCP da el punto de entrada exacto.

**El valor real no es el estado, es la secuencia**: un cribador convencional
dice qué es una acción hoy; este detector dice en qué punto de la
transición está, avanzando por subestados de FORMING a TRIGGERED
(`_evaluate`, de más a menos avanzado - un ticker se queda con el subestado
más avanzado que cumple, nunca con varios a la vez, misma regla "nadie
suma" del paquete).

**Discriminador anti-etapa-3** (Parte 13.1 exige un test explícito para
esto): una MA30 que se aplana tras una SUBIDA (una cima de etapa 3, a punto
de girar a la baja) no debe confundirse con una MA30 que se aplana tras una
BAJADA (una base de etapa 1 genuina). `_is_stage1_base_forming` exige que
la pendiente de las 8 semanas ANTERIORES a las últimas 8 ya fuera negativa -
una cima de etapa 3 llega con la pendiente anterior positiva (el resto de
la subida), nunca negativa, así que nunca cumple esta condición.

**Correcciones propias, incorporadas antes de escribir este detector** (ver
`docs/quant_methodology.md` §28.1): el secado de volumen exige
`STAGE_VOLUME_DRYUP_MIN_WEEKS` semanas *seguidas* por debajo del umbral, no
una lectura puntual (una sola semana por debajo de 0,8x es ruido, no una
señal); "RS girando" solo comprueba el cruce de la media de
`STAGE_RS_TURN_MA_WEEKS` sobre el propio RS de Mansfield - se descarta,
de momento, el "o hace un mínimo más alto" del texto original (complejidad
real por beneficio marginal sobre una serie ya de por sí ruidosa)."""

from dataclasses import dataclass

import pandas as pd

from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

STAGE_MA_WEEKS = 30
STAGE_SLOPE_LOOKBACK_WEEKS = 8
STAGE_FLAT_SLOPE_MAX_PCT = 0.005
STAGE_MIN_BASE_WEEKS = 6
STAGE_MAX_BASE_WEEKS = 52
STAGE_MAX_BASE_DEPTH_PCT = 0.35
STAGE_PRICE_NEAR_MA_PCT = 0.15
STAGE_FORMING_RANGE_MAX_PCT = 0.25
STAGE_VOLUME_DRYUP_RATIO = 0.80
# Corrección propia (§28.1): el texto original solo pedía una lectura
# puntual ("media de 5 semanas < 0,8x media de 30") - una sola semana por
# debajo de ese umbral cae dentro de la varianza normal del volumen
# semanal, no es una señal real de secado. Exigir que se sostenga varias
# semanas seguidas es lo que la convierte en una condición que de verdad
# filtra algo.
STAGE_VOLUME_DRYUP_MIN_WEEKS = 2
# Corrección propia (§28.1): el texto original nombra esta constante con
# "WEEKS" pero la describe sobre el RS de Mansfield *diario* ("a 20
# sesiones... su propia media de 10") - se implementa como 10 sesiones
# diarias, no 10 semanas, para que quede en la misma base temporal que el
# RS de 20 sesiones que suaviza. El nombre se conserva tal como lo pide el
# encargo.
STAGE_RS_TURN_MA_WEEKS = 10
STAGE_BREAKOUT_VOLUME_MULTIPLE = 2.0
STAGE_BREAKOUT_VOLUME_DAILY = 1.5
STAGE_IMMINENT_MAX_DISTANCE_ATR = 1.0
STAGE_BREAKOUT_TRIGGER_BUFFER_PCT = 0.002


def _weekly_ma30(weekly_close: pd.Series) -> pd.Series:
    return weekly_close.rolling(STAGE_MA_WEEKS).mean()


def _slope_pct(ma: pd.Series, lookback: int = STAGE_SLOPE_LOOKBACK_WEEKS) -> pd.Series:
    return ma.pct_change(periods=lookback)


def _run_length_backward(values: list[bool]) -> int:
    """Cuántos elementos seguidos, contando desde el final, son `True` -
    mismo patrón de racha-hacia-atrás que `technical_analysis._level_state_and_duration`
    ya usa para `Level.bars_in_state`."""
    run = 0
    for value in reversed(values):
        if not value:
            break
        run += 1
    return run


def _flat_base_run_weeks(slope: pd.Series) -> int:
    flat = (slope.abs() < STAGE_FLAT_SLOPE_MAX_PCT).tolist()
    return _run_length_backward(flat)


@dataclass(frozen=True, slots=True)
class _Base:
    base_high: float
    base_low: float
    depth_pct: float
    window_weeks: int


def _detect_base(weekly_close: pd.Series, run_weeks: int) -> _Base | None:
    """Parte 2.3: techo/suelo de la base sobre cierres SEMANALES de las
    últimas `min(run_weeks, STAGE_MAX_BASE_WEEKS)` semanas - `None` si la
    base es demasiado profunda para ser una base genuina (todavía una
    tendencia bajista en curso, no una acumulación)."""
    window_weeks = min(run_weeks, STAGE_MAX_BASE_WEEKS)
    window = weekly_close.iloc[-window_weeks:]
    base_high = float(window.max())
    if base_high <= 0:
        return None
    base_low = float(window.min())
    depth_pct = (base_high - base_low) / base_high
    if depth_pct > STAGE_MAX_BASE_DEPTH_PCT:
        return None
    return _Base(base_high=base_high, base_low=base_low, depth_pct=depth_pct, window_weeks=window_weeks)


def _volume_drying_up(weekly_volume: pd.Series) -> bool:
    avg5 = weekly_volume.rolling(5).mean()
    avg30 = weekly_volume.rolling(STAGE_MA_WEEKS).mean()
    ratio = (avg5 / avg30).dropna()
    if len(ratio) < STAGE_VOLUME_DRYUP_MIN_WEEKS:
        return False
    recent = ratio.iloc[-STAGE_VOLUME_DRYUP_MIN_WEEKS:]
    return bool((recent < STAGE_VOLUME_DRYUP_RATIO).all())


def _price_near_ma(price: float, ma_value: float | None) -> bool:
    if ma_value is None or pd.isna(ma_value) or ma_value == 0:
        return False
    return abs(price / ma_value - 1) <= STAGE_PRICE_NEAR_MA_PCT


def _base_evidence(base: _Base) -> dict[str, float | int | str]:
    return {
        "base_high": round(base.base_high, 4),
        "base_low": round(base.base_low, 4),
        "base_weeks": base.window_weeks,
        "base_depth_pct": round(base.depth_pct, 4),
    }


def _check_stage2_confirmed(ctx: SetupContext, base: _Base, slope: pd.Series) -> SetupMatch | None:
    slope_now = slope.iloc[-1]
    if pd.isna(slope_now) or slope_now <= 0:
        return None  # Parte 2.2: "MA30 con pendiente ya positiva" - condición dura, no opcional

    weekly_vol_avg = ctx.weekly_volume.rolling(STAGE_MA_WEEKS).mean().iloc[-1]
    weekly_confirmed = (
        float(ctx.weekly_close.iloc[-1]) > base.base_high
        and pd.notna(weekly_vol_avg)
        and weekly_vol_avg > 0
        and float(ctx.weekly_volume.iloc[-1]) >= STAGE_BREAKOUT_VOLUME_MULTIPLE * weekly_vol_avg
    )

    # Parte 2.4: "para operativa diaria, acepta también el cierre diario con
    # volumen >= 1,5x la media de 50 días, marcándolo como confirmación de
    # menor calidad" - una confirmación más débil, no una segunda condición
    # independiente que se suma a la semanal.
    daily_vol_avg = ctx.volume.rolling(50).mean().iloc[-1]
    daily_confirmed = (
        float(ctx.close.iloc[-1]) > base.base_high
        and pd.notna(daily_vol_avg)
        and daily_vol_avg > 0
        and float(ctx.volume.iloc[-1]) >= STAGE_BREAKOUT_VOLUME_DAILY * daily_vol_avg
    )

    if not (weekly_confirmed or daily_confirmed):
        return None

    if weekly_confirmed:
        timeframe, quality_note = "weekly", "cierre semanal, confirmación de calidad plena"
        bars_in_stage = _run_length_backward((ctx.weekly_close > base.base_high).tolist())
    else:
        timeframe, quality_note = "daily", "cierre diario, confirmación de menor calidad que un cierre semanal"
        bars_in_stage = _run_length_backward((ctx.close > base.base_high).tolist())

    evidence = _base_evidence(base) | {"confirmation": quality_note}
    return SetupMatch(
        family=SetupFamily.STAGE_TRANSITION,
        name="stage2_confirmed",
        label_es="Ruptura de etapa 2 confirmada",
        stage=SetupStage.TRIGGERED,
        bars_in_stage=bars_in_stage,
        timeframe=timeframe,
        trigger_price=base.base_high * (1 + STAGE_BREAKOUT_TRIGGER_BUFFER_PCT),
        trigger_condition=(
            f"cierre por encima de {base.base_high:.2f} con volumen de ruptura ({quality_note})"
        ),
        invalidation_price=base.base_low,
        invalidation_condition=f"cierre por debajo de {base.base_low:.2f} invalida la base y devuelve a etapa 4",
        evidence=evidence,
        narrative_es=(
            f"Ruptura confirmada sobre {base.base_high:.2f} tras una base de {base.window_weeks} semanas "
            f"({quality_note})."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def _check_stage2_breakout_imminent(ctx: SetupContext, base: _Base) -> SetupMatch | None:
    if ctx.atr14 is None or ctx.atr14 <= 0:
        return None
    price = float(ctx.close.iloc[-1])
    distance_atr = abs(base.base_high - price) / ctx.atr14
    if distance_atr > STAGE_IMMINENT_MAX_DISTANCE_ATR or price > base.base_high:
        return None  # ya rompió - eso es stage2_confirmed, no "inminente"

    if ctx.relative_volume is None or ctx.relative_volume < 1.0:
        return None
    # "y subiendo" - repunte reciente, no solo un nivel absoluto puntual:
    # el volumen relativo de hace 3 sesiones debía ser menor que el de hoy.
    volume_rising = len(ctx.volume) > 3 and float(ctx.volume.iloc[-1]) > float(ctx.volume.iloc[-4])

    if not volume_rising:
        return None

    evidence = _base_evidence(base) | {
        "distance_to_base_high_atr": round(distance_atr, 3),
        "relative_volume": round(ctx.relative_volume, 3),
    }
    return SetupMatch(
        family=SetupFamily.STAGE_TRANSITION,
        name="stage2_breakout_imminent",
        label_es="Ruptura de etapa 2 inminente",
        stage=SetupStage.READY,
        bars_in_stage=1,
        timeframe="daily",
        trigger_price=base.base_high * (1 + STAGE_BREAKOUT_TRIGGER_BUFFER_PCT),
        trigger_condition=f"cierre por encima de {base.base_high:.2f} con volumen de ruptura",
        invalidation_price=base.base_low,
        invalidation_condition=f"cierre por debajo de {base.base_low:.2f} invalida la base",
        evidence=evidence,
        narrative_es=(
            f"A {distance_atr:.2f} ATR del techo de la base ({base.base_high:.2f}), con el volumen "
            "empezando a repuntar. Falta el disparo."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def _mansfield_rs_turning_up(mansfield_rs_series: pd.Series | None) -> bool:
    if mansfield_rs_series is None:
        return False
    ma = mansfield_rs_series.rolling(STAGE_RS_TURN_MA_WEEKS).mean()
    valid = pd.concat([mansfield_rs_series, ma], axis=1).dropna()
    if len(valid) < 2:
        return False
    rs_now, ma_now = valid.iloc[-1, 0], valid.iloc[-1, 1]
    rs_prev, ma_prev = valid.iloc[-2, 0], valid.iloc[-2, 1]
    return bool(rs_prev <= ma_prev and rs_now > ma_now)


def _check_stage1_base_confirmed(
    ctx: SetupContext, base: _Base, run_weeks: int, weekly_ma30: pd.Series
) -> SetupMatch | None:
    price = float(ctx.close.iloc[-1])
    ma_now = weekly_ma30.iloc[-1]
    if not _price_near_ma(price, float(ma_now) if pd.notna(ma_now) else None):
        return None
    if not _volume_drying_up(ctx.weekly_volume):
        return None

    evidence = _base_evidence(base) | {"weeks_flat": run_weeks}
    rs_turning = _mansfield_rs_turning_up(ctx.mansfield_rs_series)
    if rs_turning:
        return SetupMatch(
            family=SetupFamily.STAGE_TRANSITION,
            name="stage1_rs_turning",
            label_es="Etapa 1, fuerza relativa girando al alza",
            stage=SetupStage.FORMING,
            bars_in_stage=run_weeks,
            timeframe="weekly",
            trigger_price=base.base_high * (1 + STAGE_BREAKOUT_TRIGGER_BUFFER_PCT),
            trigger_condition=f"cierre por encima de {base.base_high:.2f} con volumen de ruptura",
            invalidation_price=base.base_low,
            invalidation_condition=f"cierre por debajo de {base.base_low:.2f} invalida la base",
            evidence=evidence,
            narrative_es=(
                f"Base de {base.window_weeks} semanas con la MA30 semanal plana. El volumen se está "
                "secando y la fuerza relativa frente al mercado acaba de girar al alza - el dinero "
                "institucional puede estar entrando antes de que el precio lo muestre."
            ),
            confidence=SetupConfidence.UNVALIDATED,
        )

    return SetupMatch(
        family=SetupFamily.STAGE_TRANSITION,
        name="stage1_base_confirmed",
        label_es="Etapa 1, base confirmada",
        stage=SetupStage.FORMING,
        bars_in_stage=run_weeks,
        timeframe="weekly",
        trigger_price=base.base_high * (1 + STAGE_BREAKOUT_TRIGGER_BUFFER_PCT),
        trigger_condition=f"cierre por encima de {base.base_high:.2f} con volumen de ruptura",
        invalidation_price=base.base_low,
        invalidation_condition=f"cierre por debajo de {base.base_low:.2f} invalida la base",
        evidence=evidence,
        narrative_es=(
            f"Base de {base.window_weeks} semanas con la MA30 semanal plana y el volumen secándose. "
            "Todavía sin señal de que la fuerza relativa esté girando."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def _check_stage1_base_forming(ctx: SetupContext, slope: pd.Series) -> SetupMatch | None:
    """Parte 2.2, `stage1_base_forming`: la MA30 semanal SIGUE con pendiente
    negativa pero cada vez menos negativa - el discriminador anti-etapa-3
    vive aquí (ver el docstring del módulo)."""
    if len(slope.dropna()) < 2 * STAGE_SLOPE_LOOKBACK_WEEKS:
        return None
    slope_now = slope.iloc[-1]
    slope_prev = slope.iloc[-1 - STAGE_SLOPE_LOOKBACK_WEEKS]
    if pd.isna(slope_now) or pd.isna(slope_prev):
        return None
    # Ambas negativas (todavía en descenso) y la reciente menos negativa que
    # la anterior (desacelerando) - una cima de etapa 3 tendría `slope_prev`
    # positiva (el resto de la subida previa), nunca negativa.
    flattening = slope_prev < 0 and slope_now < 0 and slope_now > slope_prev
    if not flattening:
        return None

    price = float(ctx.close.iloc[-1])
    weekly_close = ctx.weekly_close
    weekly_ma30 = _weekly_ma30(weekly_close) if weekly_close is not None else None
    ma_value = float(weekly_ma30.iloc[-1]) if weekly_ma30 is not None and pd.notna(weekly_ma30.iloc[-1]) else None
    if not _price_near_ma(price, ma_value):
        return None

    if weekly_close is None or len(weekly_close) < STAGE_SLOPE_LOOKBACK_WEEKS:
        return None
    recent = weekly_close.iloc[-STAGE_SLOPE_LOOKBACK_WEEKS:]
    if recent.max() <= 0:
        return None
    tight_range = (recent.max() - recent.min()) / recent.max() < STAGE_FORMING_RANGE_MAX_PCT
    if not tight_range:
        return None

    return SetupMatch(
        family=SetupFamily.STAGE_TRANSITION,
        name="stage1_base_forming",
        label_es="Etapa 1, base formándose",
        stage=SetupStage.FORMING,
        bars_in_stage=STAGE_SLOPE_LOOKBACK_WEEKS,
        timeframe="weekly",
        trigger_price=None,
        trigger_condition="todavía no hay techo de base definido - falta confirmar la etapa 1",
        invalidation_price=None,
        invalidation_condition="",
        evidence={
            "slope_now_pct": round(float(slope_now) * 100, 3),
            "slope_prev_pct": round(float(slope_prev) * 100, 3),
        },
        narrative_es=(
            "La caída se está frenando: la MA30 semanal sigue bajando pero cada vez más despacio, "
            "y el precio se mantiene cerca de ella en un rango estrecho. Todavía no hay una base "
            "confirmada - falta que la MA30 termine de aplanarse."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def _base_candidates(weekly_close: pd.Series, slope: pd.Series) -> list[tuple[_Base, int]]:
    """Candidatos de base a evaluar, de más a menos reciente: la racha plana
    tal cual está midiéndose hoy, y - si la de hoy ya no llega a
    `STAGE_MIN_BASE_WEEKS` - la que había justo una semana antes. Sin esto,
    una ruptura genuina (`stage2_confirmed`) sería indetectable: el propio
    cierre de esta semana ya mueve la pendiente de la MA30 fuera del rango
    "plano", así que medir la racha plana sobre la pendiente *de hoy* la
    encontraría siempre en 0 justo en el momento en que la ruptura ocurre -
    la base hay que medirla desde antes de eso, no desde una pendiente que
    ya refleja la ruptura misma."""
    candidates: list[tuple[_Base, int]] = []
    run_weeks = _flat_base_run_weeks(slope)
    if run_weeks >= STAGE_MIN_BASE_WEEKS:
        base = _detect_base(weekly_close, run_weeks)
        if base is not None:
            candidates.append((base, run_weeks))
    if len(slope) > 1:
        prior_run_weeks = _flat_base_run_weeks(slope.iloc[:-1])
        if prior_run_weeks >= STAGE_MIN_BASE_WEEKS:
            prior_base = _detect_base(weekly_close.iloc[:-1], prior_run_weeks)
            if prior_base is not None:
                candidates.append((prior_base, prior_run_weeks))
    return candidates


def detect(ctx: SetupContext) -> list[SetupMatch]:
    if ctx.weekly_close is None or ctx.weekly_volume is None:
        return []
    if len(ctx.weekly_close) < STAGE_MA_WEEKS + 2 * STAGE_SLOPE_LOOKBACK_WEEKS:
        return []

    weekly_ma30 = _weekly_ma30(ctx.weekly_close)
    slope = _slope_pct(weekly_ma30)
    if slope.dropna().empty:
        return []

    for base, run_weeks in _base_candidates(ctx.weekly_close, slope):
        match = (
            _check_stage2_confirmed(ctx, base, slope)
            or _check_stage2_breakout_imminent(ctx, base)
            or _check_stage1_base_confirmed(ctx, base, run_weeks, weekly_ma30)
        )
        if match is not None:
            return [match]

    forming = _check_stage1_base_forming(ctx, slope)
    return [forming] if forming is not None else []
