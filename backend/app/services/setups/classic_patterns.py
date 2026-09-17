"""Parte 5 del encargo: patrones clásicos - taza con asa, doble suelo,
triángulos, hombro-cabeza-hombro. Ver `setups/__init__.py` para el
estándar de evidencia completo de esta familia (Lo, Mamaysky y Wang 2000):
se detectan porque el propietario los quiere ver, no porque exista
evidencia sólida de que operarlos sea rentable - por eso el tratamiento es
asimétrico, `CLASSIC_PATTERNS_CAN_TRIGGER` es la única lista blanca que
puede emitir `trigger_price`, el resto siempre lo deja en `None`.

A diferencia de toda otra familia de esta biblioteca (que devuelve como
mucho un `SetupMatch`, una progresión de UN patrón), esta familia agrupa
VARIAS FORMAS DISTINTAS bajo el mismo `SetupFamily.CLASSIC_PATTERN` -
`detect` puede devolver más de un match a la vez (p. ej. un doble suelo Y
un triángulo detectados simultáneamente son dos patrones genuinamente
distintos, no dos etapas del mismo) - la elección entre familias sigue
viviendo en `arbitration.py`, esto solo agrupa formas dentro de una misma
familia.

Las cuatro familias completas de la Parte 5: doble suelo y triángulos (los
dos que reutilizan infraestructura ya construida - `technical_analysis.indexed_fractal_pivots`,
que `vcp.py` ya usa, y `technical_analysis.linear_regression_fit` con
posiciones `x` explícitas, que `channel.py` ya usa sobre una ventana densa
de cierres), más taza con asa y hombro-cabeza-hombro (dos variantes cada
una - cinco clasificaciones de triángulo, invertido/normal de
hombro-cabeza-hombro), añadidas después sobre la misma base. Todo el
módulo trabaja sobre barras diarias sin remuestrear, incluida la taza (el
encargo la describe "en semanas" - conversión literal vía 5 sesiones/semana,
ver su propia sección - para no introducir una costura semanal/diaria justo
donde más importa: el asa, que el encargo sí da "en sesiones")."""

import pandas as pd

from app.services import technical_analysis as ta
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

CLASSIC_PATTERNS_CAN_TRIGGER = frozenset({"cup_with_handle", "double_bottom", "ascending_triangle"})

# --- Doble suelo (Parte 5.3) --------------------------------------------------

DOUBLE_BOTTOM_LOOKBACK_BARS = 120
DOUBLE_BOTTOM_MIN_SEPARATION = 15
DOUBLE_BOTTOM_MAX_SEPARATION = 90
DOUBLE_BOTTOM_MAX_DIFF_PCT = 0.04
DOUBLE_BOTTOM_MIN_INTERMEDIATE_HIGH_PCT = 0.08
DOUBLE_BOTTOM_TRIGGER_BUFFER_PCT = 0.002


def _detect_double_bottom(ctx: SetupContext) -> SetupMatch | None:
    close = ctx.close.iloc[-DOUBLE_BOTTOM_LOOKBACK_BARS:].reset_index(drop=True)
    volume = ctx.volume.iloc[-DOUBLE_BOTTOM_LOOKBACK_BARS:].reset_index(drop=True)
    lows = ta.indexed_fractal_pivots(close, 3, 3, "low")
    if len(lows) < 2:
        return None
    (l1_idx, l1_price), (l2_idx, l2_price) = lows[-2], lows[-1]

    separation = l2_idx - l1_idx
    if not (DOUBLE_BOTTOM_MIN_SEPARATION <= separation <= DOUBLE_BOTTOM_MAX_SEPARATION):
        return None
    diff_pct = abs(l1_price - l2_price) / max(l1_price, l2_price)
    if diff_pct > DOUBLE_BOTTOM_MAX_DIFF_PCT:
        return None

    highs = ta.indexed_fractal_pivots(close, 3, 3, "high")
    between = [(i, p) for i, p in highs if l1_idx < i < l2_idx]
    if not between:
        return None
    _, mid_price = max(between, key=lambda t: t[1])
    lows_avg = (l1_price + l2_price) / 2
    if lows_avg <= 0 or (mid_price - lows_avg) / lows_avg < DOUBLE_BOTTOM_MIN_INTERMEDIATE_HIGH_PCT:
        return None

    # "El segundo mínimo con menor volumen que el primero" - es lo que
    # distingue un doble suelo real de una caída en dos tramos.
    if float(volume.iloc[l2_idx]) >= float(volume.iloc[l1_idx]):
        return None

    trigger_price = mid_price * (1 + DOUBLE_BOTTOM_TRIGGER_BUFFER_PCT)
    invalidation_price = min(l1_price, l2_price)
    price = float(ctx.close.iloc[-1])
    stage = SetupStage.TRIGGERED if price > trigger_price else SetupStage.READY

    return SetupMatch(
        family=SetupFamily.CLASSIC_PATTERN,
        name="double_bottom",
        label_es="Doble suelo",
        stage=stage,
        bars_in_stage=len(close) - 1 - l2_idx,  # posiciones relativas a la misma ventana recortada
        timeframe="daily",
        trigger_price=trigger_price,
        trigger_condition=f"cierre por encima del máximo intermedio ({mid_price:.2f})",
        invalidation_price=invalidation_price,
        invalidation_condition=f"cierre por debajo de {invalidation_price:.2f} invalida el patrón",
        evidence={
            "low1": round(l1_price, 4), "low2": round(l2_price, 4), "separation_bars": separation,
            "diff_pct": round(diff_pct, 4), "intermediate_high": round(mid_price, 4),
        },
        narrative_es=(
            f"Doble suelo en {separation} sesiones ({l1_price:.2f} y {l2_price:.2f}, diferencia "
            f"{diff_pct * 100:.1f}%), segundo mínimo con menos volumen que el primero."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


# --- Triángulos (Parte 5.4) ---------------------------------------------------

TRIANGLE_LOOKBACK_BARS = 90
TRIANGLE_PIVOT_LEFT_RIGHT = 3
TRIANGLE_MIN_PIVOTS_PER_SIDE = 3
TRIANGLE_MAX_BARS_TO_APEX = 60
# Umbral de "pendiente prácticamente plana", normalizado por ATR (el mismo
# idioma que el resto del proyecto usa para comparar distancias entre
# activos de volatilidad distinta) - no hay un umbral numérico literal en
# el encargo para "qué tan plano es plano" en un triángulo, a diferencia de
# casi todos los demás umbrales de esta biblioteca.
TRIANGLE_FLAT_SLOPE_ATR_FRACTION = 0.05

_TRIANGLE_LABELS_ES = {
    "ascending_triangle": "Triángulo ascendente",
    "descending_triangle": "Triángulo descendente",
    "symmetrical_triangle": "Triángulo simétrico",
    "rising_wedge": "Cuña ascendente",
    "falling_wedge": "Cuña descendente",
}


def _classify_triangle(high_slope: float, low_slope: float, flat: float) -> str | None:
    high_flat, low_flat = abs(high_slope) < flat, abs(low_slope) < flat
    high_up, high_down = high_slope > flat, high_slope < -flat
    low_up, low_down = low_slope > flat, low_slope < -flat

    if high_flat and low_up:
        return "ascending_triangle"
    if high_down and low_flat:
        return "descending_triangle"
    if high_down and low_up:
        return "symmetrical_triangle"
    if high_up and low_up:
        return "rising_wedge"
    if high_down and low_down:
        return "falling_wedge"
    return None


def _detect_triangle(ctx: SetupContext) -> SetupMatch | None:
    # `bars_in_stage` es una diferencia entre dos posiciones DENTRO de esta
    # misma ventana recortada - no hace falta volver a la posición absoluta
    # en `ctx.close` (a diferencia de `vcp.py`, que sí persiste un índice
    # fuera de esta función).
    close = ctx.close.iloc[-TRIANGLE_LOOKBACK_BARS:].reset_index(drop=True)
    highs = ta.indexed_fractal_pivots(close, TRIANGLE_PIVOT_LEFT_RIGHT, TRIANGLE_PIVOT_LEFT_RIGHT, "high")
    lows = ta.indexed_fractal_pivots(close, TRIANGLE_PIVOT_LEFT_RIGHT, TRIANGLE_PIVOT_LEFT_RIGHT, "low")
    if len(highs) < TRIANGLE_MIN_PIVOTS_PER_SIDE or len(lows) < TRIANGLE_MIN_PIVOTS_PER_SIDE:
        return None

    high_fit = ta.linear_regression_fit(pd.Series([p for _, p in highs]), x=[float(i) for i, _ in highs])
    low_fit = ta.linear_regression_fit(pd.Series([p for _, p in lows]), x=[float(i) for i, _ in lows])
    if high_fit is None or low_fit is None:
        return None

    if ctx.atr14 is None or ctx.atr14 <= 0:
        return None
    flat_threshold = TRIANGLE_FLAT_SLOPE_ATR_FRACTION * ctx.atr14
    kind = _classify_triangle(high_fit.slope, low_fit.slope, flat_threshold)
    if kind is None:
        return None

    # Punto de convergencia: dónde se cruzan las dos rectas - fuera de una
    # ventana futura razonable, son "dos rectas cualquiera", no un triángulo.
    slope_diff = high_fit.slope - low_fit.slope
    if slope_diff == 0:
        return None
    apex_x = (low_fit.intercept - high_fit.intercept) / slope_diff
    last_pivot_x = max(highs[-1][0], lows[-1][0])
    bars_to_apex = apex_x - last_pivot_x
    if not (0 < bars_to_apex <= TRIANGLE_MAX_BARS_TO_APEX):
        return None

    can_trigger = kind in CLASSIC_PATTERNS_CAN_TRIGGER
    resistance_now = high_fit.slope * (len(close) - 1) + high_fit.intercept
    trigger_price = resistance_now if can_trigger else None
    support_now = low_fit.slope * (len(close) - 1) + low_fit.intercept

    return SetupMatch(
        family=SetupFamily.CLASSIC_PATTERN,
        name=kind,
        label_es=_TRIANGLE_LABELS_ES[kind],
        stage=SetupStage.READY if can_trigger else SetupStage.FORMING,
        bars_in_stage=len(close) - 1 - last_pivot_x,
        timeframe="daily",
        trigger_price=trigger_price,
        trigger_condition=(
            f"cierre por encima de {resistance_now:.2f} (techo del triángulo)" if can_trigger else ""
        ),
        invalidation_price=support_now if can_trigger else None,
        invalidation_condition=(
            f"cierre por debajo de {support_now:.2f} invalida el patrón" if can_trigger else ""
        ),
        evidence={
            "high_slope": round(high_fit.slope, 5), "low_slope": round(low_fit.slope, 5),
            "bars_to_apex": round(bars_to_apex, 1), "n_highs": len(highs), "n_lows": len(lows),
        },
        narrative_es=(
            f"{_TRIANGLE_LABELS_ES[kind]}, convergiendo en ~{bars_to_apex:.0f} sesiones."
            + ("" if can_trigger else " Contexto - no dispara por sí solo.")
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


# --- Taza con asa (Parte 5.2) --------------------------------------------

CUP_TRADING_DAYS_PER_WEEK = 5
# El encargo da la duración de la taza "en semanas" (canon de O'Neil), pero
# igual que el resto de esta biblioteca (VCP, ruptura, canal, doble suelo,
# triángulo) este detector trabaja sobre barras diarias sin remuestrear -
# evita una costura semanal/diaria justo donde más importa: el asa, que el
# propio encargo da "en sesiones", no en semanas. Conversión literal
# semanas -> sesiones vía 5 sesiones/semana.
CUP_MIN_DURATION_BARS = 7 * CUP_TRADING_DAYS_PER_WEEK  # 35
CUP_MAX_DURATION_BARS = 65 * CUP_TRADING_DAYS_PER_WEEK  # 325
CUP_MIN_DEPTH_PCT = 0.12
CUP_TYPICAL_MAX_DEPTH_PCT = 0.33
CUP_MAX_DEPTH_PCT = 0.50
# Forma en U, no en V: el "tramo bajo" (el quinto inferior del rango de la
# taza, medido sobre cierres para no depender de una sola mecha) debe
# abarcar al menos el 30% del ancho total de la taza - el encargo pide la
# regla ("dura al menos el 30% de su anchura total") pero no fija el
# algoritmo exacto de "tramo bajo"; esta es la lectura de esta implementación.
#
# Ojo con el umbral de profundidad de la zona baja: para CUALQUIER rampa
# lineal (declive recta + recuperación recta - la "V" más simple posible),
# la fracción de ancho que cae dentro del f% inferior de la altura es, por
# semejanza de triángulos, aproximadamente f mismo - independiente de lo
# ancha o profunda que sea la taza. Verificado con un script antes de fijar
# este valor (quant_methodology.md): con f=0,33 casi igual al mínimo de
# 0,30 exigido, una V perfectamente recta pasaba el filtro por pura
# geometría, sin ninguna base plana real. f=0,20 deja un margen genuino
# (una V recta da ~0,20-0,22, muy por debajo del 0,30 exigido) sin dejar de
# aceptar una taza real con una base visiblemente más plana que una recta.
CUP_LOW_ZONE_DEPTH_FRACTION = 0.20
CUP_MIN_LOW_STRETCH_FRACTION = 0.30

HANDLE_MIN_DURATION_BARS = 5
# Sin número literal en el encargo para el máximo del asa (a diferencia de
# su mínimo, "≥ 5 sesiones") - mismo tratamiento que
# TRIANGLE_FLAT_SLOPE_ATR_FRACTION: elección propia documentada, no un
# umbral del encargo. ~5 semanas, el orden de magnitud habitual de un asa.
HANDLE_MAX_DURATION_BARS = 25
HANDLE_MIN_DEPTH_PCT = 0.08
HANDLE_TYPICAL_MAX_DEPTH_PCT = 0.12
HANDLE_MAX_DEPTH_PCT = 0.15
CUP_HANDLE_TRIGGER_BUFFER_PCT = 0.002
CUP_HANDLE_LOOKBACK_BARS = CUP_MAX_DURATION_BARS + HANDLE_MAX_DURATION_BARS


def _handle_volume_declining(volume: pd.Series) -> bool:
    mid = len(volume) // 2
    first_half, second_half = volume.iloc[:mid], volume.iloc[mid:]
    if first_half.empty or second_half.empty:
        return False
    return bool(second_half.mean() < first_half.mean())


def _detect_cup_with_handle(ctx: SetupContext) -> SetupMatch | None:
    full_close = ctx.close.iloc[-CUP_HANDLE_LOOKBACK_BARS:].reset_index(drop=True)
    full_volume = ctx.volume.iloc[-CUP_HANDLE_LOOKBACK_BARS:].reset_index(drop=True)
    if len(full_close) < CUP_MIN_DURATION_BARS + HANDLE_MIN_DURATION_BARS + 1:
        return None

    # Toda la geometría (labios, fondo, forma del asa) se mide EXCLUYENDO
    # la barra de hoy a propósito - la misma trampa autorreferencial que
    # `breakout._darvas_box` ya documenta: si hoy se incluyera, una ruptura
    # del asa simplemente "ensancharía el asa hasta hoy" (el máximo de la
    # ventana pasaría a ser la propia barra de ruptura) en vez de poder
    # leerse como una ruptura de una estructura que ya existía ANTES de
    # hoy - `full_close`/`full_volume` (con hoy) solo vuelven a usarse al
    # final, para decidir READY vs TRIGGERED contra el pivote ya calculado
    # sin la barra de hoy. Encontrado con el script de verificación de este
    # detector antes de fijar los tests (quant_methodology.md).
    close = full_close.iloc[:-1]
    volume = full_volume.iloc[:-1]

    # Labio derecho: el máximo de las últimas HANDLE_MAX_DURATION_BARS
    # sesiones - si el asa todavía no lleva tantas sesiones formándose, el
    # máximo de esta ventana corta YA ES ese labio, sin necesitar un pivote
    # confirmado (que llegaría demasiado tarde: justo cuando el asa es más
    # accionable). Misma lógica que `pullback._last_impulse`.
    handle_window = close.iloc[-HANDLE_MAX_DURATION_BARS:]
    right_lip_idx = int(handle_window.idxmax())
    right_lip_price = float(close.iloc[right_lip_idx])
    handle_duration_bars = len(close) - 1 - right_lip_idx
    if not (HANDLE_MIN_DURATION_BARS <= handle_duration_bars <= HANDLE_MAX_DURATION_BARS):
        return None

    pre_handle = close.iloc[:right_lip_idx + 1]
    bottom_idx = int(pre_handle.idxmin())
    bottom_price = float(close.iloc[bottom_idx])
    if bottom_idx == 0:
        return None  # no hay barras antes del fondo para buscar el labio izquierdo
    left_lip_idx = int(close.iloc[:bottom_idx].idxmax())
    left_lip_price = float(close.iloc[left_lip_idx])

    cup_duration_bars = right_lip_idx - left_lip_idx
    if not (CUP_MIN_DURATION_BARS <= cup_duration_bars <= CUP_MAX_DURATION_BARS):
        return None
    if left_lip_price <= 0:
        return None
    cup_depth_pct = (left_lip_price - bottom_price) / left_lip_price
    if not (CUP_MIN_DEPTH_PCT <= cup_depth_pct <= CUP_MAX_DEPTH_PCT):
        return None

    low_zone_threshold = bottom_price + CUP_LOW_ZONE_DEPTH_FRACTION * (left_lip_price - bottom_price)
    cup_slice = close.iloc[left_lip_idx:right_lip_idx + 1]
    in_low_zone = cup_slice[cup_slice <= low_zone_threshold]
    if in_low_zone.empty:
        return None
    low_stretch_bars = int(in_low_zone.index.max() - in_low_zone.index.min()) + 1
    if low_stretch_bars / cup_duration_bars < CUP_MIN_LOW_STRETCH_FRACTION:
        return None  # forma en V, no en U

    cup_midpoint = bottom_price + 0.5 * (left_lip_price - bottom_price)
    handle_close = close.iloc[right_lip_idx + 1:]
    handle_volume = volume.iloc[right_lip_idx + 1:]
    handle_low = float(handle_close.min())
    if handle_low < cup_midpoint:
        return None  # el asa se formó en la mitad inferior de la taza

    handle_depth_pct = (right_lip_price - handle_low) / right_lip_price
    if not (HANDLE_MIN_DEPTH_PCT <= handle_depth_pct <= HANDLE_MAX_DEPTH_PCT):
        return None

    handle_fit = ta.linear_regression_fit(handle_close)
    if handle_fit is None or handle_fit.slope >= 0:
        return None  # un asa que sube no es un asa
    if not _handle_volume_declining(handle_volume):
        return None

    trigger_price = right_lip_price * (1 + CUP_HANDLE_TRIGGER_BUFFER_PCT)
    invalidation_price = handle_low
    price = float(full_close.iloc[-1])  # la única lectura que sí incluye la barra de hoy
    stage = SetupStage.TRIGGERED if price > trigger_price else SetupStage.READY
    cup_quality = "profunda" if cup_depth_pct > CUP_TYPICAL_MAX_DEPTH_PCT else "típica"

    return SetupMatch(
        family=SetupFamily.CLASSIC_PATTERN,
        name="cup_with_handle",
        label_es="Taza con asa",
        stage=stage,
        bars_in_stage=len(full_close) - 1 - right_lip_idx,
        timeframe="daily",
        trigger_price=trigger_price,
        trigger_condition=f"cierre por encima de {trigger_price:.2f} (máximo del asa + 0,2%)",
        invalidation_price=invalidation_price,
        invalidation_condition=f"cierre por debajo de {invalidation_price:.2f} rompe el asa",
        evidence={
            "left_lip": round(left_lip_price, 4), "bottom": round(bottom_price, 4),
            "right_lip": round(right_lip_price, 4), "cup_depth_pct": round(cup_depth_pct, 4),
            "cup_duration_bars": cup_duration_bars, "cup_quality": cup_quality,
            "handle_depth_pct": round(handle_depth_pct, 4), "handle_duration_bars": handle_duration_bars,
        },
        narrative_es=(
            f"Taza con asa {cup_quality}: profundidad {cup_depth_pct * 100:.0f}% en {cup_duration_bars} "
            f"sesiones, asa de {handle_duration_bars} sesiones ({handle_depth_pct * 100:.0f}%) derivando "
            f"a la baja con volumen descendente."
        ),
        # El encargo (5.2) pide MEASURED/THIN según un umbral de muestra
        # histórica para esta forma en concreto - eso exige el replay de la
        # Parte 10 (`setup_replay.py`, todavía no construido). Hasta
        # entonces, UNVALIDATED - la misma regla no negociable de
        # `types.SetupConfidence` para cualquier detector nuevo.
        confidence=SetupConfidence.UNVALIDATED,
    )


# --- Hombro-cabeza-hombro (Parte 5.5) -------------------------------------

HEAD_SHOULDERS_LOOKBACK_BARS = 150
HEAD_SHOULDERS_PIVOT_LEFT_RIGHT = 3
HEAD_SHOULDERS_MIN_HEAD_PROMINENCE_PCT = 0.03
HEAD_SHOULDERS_MAX_SHOULDER_ASYMMETRY_PCT = 0.05
# "Pendiente menor al 10% de la altura del patrón" - leído como el cambio
# TOTAL de la clavicular entre sus dos puntos (no una pendiente por barra,
# que no sería dimensionalmente comparable contra una altura en precio sin
# normalizar por tiempo) respecto a la altura cabeza-clavicular. Elección
# de lectura propia, documentada como tal.
HEAD_SHOULDERS_MAX_NECKLINE_DRIFT_FRACTION = 0.10

_HEAD_SHOULDERS_LABELS_ES = {
    "head_shoulders_inverse": "Hombro-cabeza-hombro invertido",
    "head_shoulders_top": "Hombro-cabeza-hombro",
}


def _detect_head_shoulders(ctx: SetupContext, kind: str) -> SetupMatch | None:
    """Ninguna de las dos formas dispara jamás (`trigger_price` fijo a
    `None`) - a diferencia de los triángulos, aquí no hace falta pasar por
    `CLASSIC_PATTERNS_CAN_TRIGGER`: el encargo es explícito en que NINGUNA
    de las dos genera una entrada (la invertida por solaparse con la
    transición de etapa 1 a 2, ya mejor especificada; la normal por ser
    puramente una bandera de aviso) - no es un caso dinámico como la
    clasificación de triángulos, donde una sola función puede producir la
    única forma que sí dispara."""
    close = ctx.close.iloc[-HEAD_SHOULDERS_LOOKBACK_BARS:].reset_index(drop=True)
    is_inverse = kind == "head_shoulders_inverse"
    extreme_kind = "low" if is_inverse else "high"
    neckline_kind = "high" if is_inverse else "low"

    extremes = ta.indexed_fractal_pivots(
        close, HEAD_SHOULDERS_PIVOT_LEFT_RIGHT, HEAD_SHOULDERS_PIVOT_LEFT_RIGHT, extreme_kind
    )
    if len(extremes) < 3:
        return None
    (ls_idx, ls_price), (h_idx, h_price), (rs_idx, rs_price) = extremes[-3:]

    head_beats_shoulders = (
        (h_price < ls_price and h_price < rs_price) if is_inverse
        else (h_price > ls_price and h_price > rs_price)
    )
    if not head_beats_shoulders:
        return None

    if is_inverse:
        prominence_ls = (ls_price - h_price) / ls_price
        prominence_rs = (rs_price - h_price) / rs_price
    else:
        prominence_ls = (h_price - ls_price) / ls_price
        prominence_rs = (h_price - rs_price) / rs_price
    if (
        prominence_ls < HEAD_SHOULDERS_MIN_HEAD_PROMINENCE_PCT
        or prominence_rs < HEAD_SHOULDERS_MIN_HEAD_PROMINENCE_PCT
    ):
        return None

    shoulder_diff_pct = abs(ls_price - rs_price) / max(ls_price, rs_price)
    if shoulder_diff_pct > HEAD_SHOULDERS_MAX_SHOULDER_ASYMMETRY_PCT:
        return None

    left_span = close.iloc[ls_idx:h_idx + 1]
    right_span = close.iloc[h_idx:rs_idx + 1]
    n1_idx = int(left_span.idxmax() if neckline_kind == "high" else left_span.idxmin())
    n2_idx = int(right_span.idxmax() if neckline_kind == "high" else right_span.idxmin())
    if n2_idx == n1_idx:
        return None
    n1_price, n2_price = float(close.iloc[n1_idx]), float(close.iloc[n2_idx])

    pattern_height = abs(h_price - (n1_price + n2_price) / 2)
    if pattern_height <= 0:
        return None
    neckline_drift = abs(n2_price - n1_price)
    if neckline_drift > HEAD_SHOULDERS_MAX_NECKLINE_DRIFT_FRACTION * pattern_height:
        return None

    neckline_now = n2_price
    price = float(close.iloc[-1])
    broken = price > neckline_now if is_inverse else price < neckline_now
    stage = SetupStage.TRIGGERED if broken else SetupStage.FORMING
    invalidation_price = h_price
    bars_in_stage = len(close) - 1 - rs_idx

    action_es = (
        "Contexto alcista de apoyo - no dispara por sí solo (se solapa con la transición de etapa 1 a 2)."
        if is_inverse
        else "Bandera de evitar / aviso en posición abierta - nunca genera una entrada."
    )
    break_es = "rompió" if broken else "todavía no rompe"
    side_es = "por debajo" if is_inverse else "por encima"

    return SetupMatch(
        family=SetupFamily.CLASSIC_PATTERN,
        name=kind,
        label_es=_HEAD_SHOULDERS_LABELS_ES[kind],
        stage=stage,
        bars_in_stage=bars_in_stage,
        timeframe="daily",
        trigger_price=None,
        trigger_condition="",
        invalidation_price=invalidation_price,
        invalidation_condition=f"cierre {side_es} de {invalidation_price:.2f} invalida el patrón",
        evidence={
            "left_shoulder": round(ls_price, 4), "head": round(h_price, 4), "right_shoulder": round(rs_price, 4),
            "neckline": round(neckline_now, 4), "shoulder_diff_pct": round(shoulder_diff_pct, 4),
        },
        narrative_es=(
            f"{_HEAD_SHOULDERS_LABELS_ES[kind]}: hombros a {ls_price:.2f}/{rs_price:.2f}, cabeza en "
            f"{h_price:.2f}, clavicular en {neckline_now:.2f} ({break_es}). {action_es}"
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def detect(ctx: SetupContext) -> list[SetupMatch]:
    matches = [
        _detect_double_bottom(ctx),
        _detect_triangle(ctx),
        _detect_cup_with_handle(ctx),
        _detect_head_shoulders(ctx, "head_shoulders_inverse"),
        _detect_head_shoulders(ctx, "head_shoulders_top"),
    ]
    return [m for m in matches if m is not None]
