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

Este módulo, primera entrega: doble suelo y triángulos (los dos que
reutilizan infraestructura ya construida - `technical_analysis.indexed_fractal_pivots`,
que `vcp.py` ya usa, y `technical_analysis.linear_regression_fit` con
posiciones `x` explícitas, que `channel.py` ya usa sobre una ventana densa
de cierres). Taza con asa y hombro-cabeza-hombro llegan en una fase
posterior."""

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


def detect(ctx: SetupContext) -> list[SetupMatch]:
    matches = [_detect_double_bottom(ctx), _detect_triangle(ctx)]
    return [m for m in matches if m is not None]
