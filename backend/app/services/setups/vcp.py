"""Parte 3 del encargo: VCP (Volatility Contraction Pattern), el patrón
propio de Minervini que concreta la anticipación de la etapa 1→2 -
`stage_transition.py` (Parte 2) da el contexto de Weinstein, este detector
da el punto de entrada exacto que se forma al final de esa etapa.

**La familia genuinamente cara de construir, documentado desde el propio
plan de fases (§28.1)**: necesita la SECUENCIA cronológica de pivotes
(máximo→mínimo→máximo...), no solo sus precios - `technical_analysis._fractal_pivots`
(la primitiva compartida, usada por `detect_levels`) descarta la posición
de cada pivote porque `detect_levels` no la necesita. `_indexed_fractal_pivots`
reimplementa aquí la MISMA definición exacta de pivote (no una nueva regla),
solo conservando también su índice - "no repitas piezas" se cumple a nivel
de lógica, no de firma de función, cuando la firma compartida no puede dar
lo que este detector necesita sin cambiar a todos los demás consumidores.

**Correcciones propias, incorporadas antes de escribir este detector** (ver
quant_methodology.md §28.1): la tolerancia de "contracción decreciente" del
texto original (15%) se baja a `VCP_CONTRACTION_TOLERANCE=0,08` - con 15%,
una secuencia 20%→22%→18% pasaba como "decreciente" pese a que el segundo
tramo es más grande que el primero. Misma tolerancia corregida para el
volumen decreciente."""

import numpy as np
import pandas as pd

from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

VCP_PIVOT_LEFT_RIGHT = 3
VCP_LOOKBACK_BARS = 90
VCP_MIN_CONTRACTIONS = 2
VCP_MAX_CONTRACTIONS = 6
VCP_CONTRACTION_TOLERANCE = 0.08  # corrección propia - el texto original pedía 0.15
VCP_FINAL_CONTRACTION_MAX_PCT = 0.10
VCP_MIN_BASE_DAYS = 30
VCP_MAX_BASE_DAYS = 90
VCP_READY_MAX_DISTANCE_ATR = 1.5
VCP_TRIGGER_VOLUME_MULTIPLE = 1.4
VCP_TRIGGER_VOLUME_WINDOW = 50
VCP_FAILED_LOOKBACK_DAYS = 3
VCP_TRIGGER_BUFFER_PCT = 0.002


def _indexed_fractal_pivots(series: pd.Series, kind: str) -> list[tuple[int, float]]:
    values = series.to_numpy()
    left = right = VCP_PIVOT_LEFT_RIGHT
    pivots = []
    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        center = values[i]
        if kind == "high" and center == window.max() and np.count_nonzero(window == center) == 1:
            pivots.append((i, float(center)))
        elif kind == "low" and center == window.min() and np.count_nonzero(window == center) == 1:
            pivots.append((i, float(center)))
    return pivots


def _alternating_pivots(
    highs: list[tuple[int, float]], lows: list[tuple[int, float]]
) -> list[tuple[int, str, float]]:
    """Fusiona altos y bajos en una secuencia cronológica que alterna de
    verdad - dos pivotes del mismo lado seguidos (sin uno del lado
    contrario entre medias) se colapsan al más extremo de los dos, que es
    el swing real."""
    combined = sorted(
        [(i, "high", p) for i, p in highs] + [(i, "low", p) for i, p in lows], key=lambda t: t[0]
    )
    if not combined:
        return []
    result: list[tuple[int, str, float]] = [combined[0]]
    for item in combined[1:]:
        idx, kind, price = item
        _, last_kind, last_price = result[-1]
        if kind == last_kind:
            if (kind == "high" and price > last_price) or (kind == "low" and price < last_price):
                result[-1] = item
        else:
            result.append(item)
    return result


def _contractions(alternating: list[tuple[int, str, float]]) -> list[tuple[float, float, int, int]]:
    """`(pivot_high, pivot_low, high_idx, low_idx)` por cada tramo
    High->Low sucesivo - una contracción es, literalmente, ese tramo."""
    return [
        (a[2], b[2], a[0], b[0])
        for a, b in zip(alternating, alternating[1:], strict=False)
        if a[1] == "high" and b[1] == "low"
    ]


def _depths_decreasing(depths: list[float]) -> bool:
    return all(depths[i + 1] < depths[i] * (1 + VCP_CONTRACTION_TOLERANCE) for i in range(len(depths) - 1))


def _mean_volume_between(volume: pd.Series, start_idx: int, end_idx: int) -> float:
    return float(volume.iloc[start_idx : end_idx + 1].mean())


def detect(ctx: SetupContext) -> list[SetupMatch]:
    close = ctx.close.iloc[-VCP_LOOKBACK_BARS:]
    if len(close) < VCP_MIN_BASE_DAYS:
        return []
    offset = len(ctx.close) - len(close)
    volume = ctx.volume.iloc[-VCP_LOOKBACK_BARS:].reset_index(drop=True)
    close = close.reset_index(drop=True)

    highs = _indexed_fractal_pivots(close, "high")
    lows = _indexed_fractal_pivots(close, "low")
    alternating = _alternating_pivots(highs, lows)
    contractions = _contractions(alternating)
    if not (VCP_MIN_CONTRACTIONS <= len(contractions) <= VCP_MAX_CONTRACTIONS):
        return []

    depths = [(high - low) / high for high, low, _, _ in contractions if high > 0]
    if len(depths) != len(contractions):
        return []
    if not _depths_decreasing(depths):
        return []

    volumes = [_mean_volume_between(volume, hi, li) for _, _, hi, li in contractions]
    if not _depths_decreasing([v / volumes[0] if volumes[0] > 0 else 1.0 for v in volumes]):
        return []

    base_days = contractions[-1][3] - contractions[0][2]
    if not (VCP_MIN_BASE_DAYS <= base_days <= VCP_MAX_BASE_DAYS):
        return []

    pivot_price = contractions[-1][0]  # el máximo de la última contracción
    final_depth = depths[-1]
    price = float(ctx.close.iloc[-1])

    evidence: dict[str, float | int | str] = {
        "n_contractions": len(contractions),
        "depths_pct": [round(d, 4) for d in depths],
        "volume_ratios": [round(v / volumes[0], 3) if volumes[0] > 0 else None for v in volumes],
        "pivot_price": round(pivot_price, 4),
        "base_days": base_days,
    }
    trigger_price = pivot_price * (1 + VCP_TRIGGER_BUFFER_PCT)
    invalidation_price = contractions[-1][1]  # el mínimo de la última contracción

    # vcp_triggered: cierre por encima del pivote con volumen de ruptura.
    trigger_vol_avg = ctx.volume.rolling(VCP_TRIGGER_VOLUME_WINDOW).mean().iloc[-1]
    triggered = (
        price > trigger_price
        and pd.notna(trigger_vol_avg)
        and trigger_vol_avg > 0
        and float(ctx.volume.iloc[-1]) >= VCP_TRIGGER_VOLUME_MULTIPLE * trigger_vol_avg
    )
    # vcp_failed: disparó y volvió por debajo del pivote en los últimos
    # `VCP_FAILED_LOOKBACK_DAYS` cierres - sin exigir volumen (ya falló,
    # el volumen del intento ya no importa).
    recent_closes = ctx.close.iloc[-(VCP_FAILED_LOOKBACK_DAYS + 1) :]
    failed = len(recent_closes) > 1 and recent_closes.iloc[:-1].max() > trigger_price and price < pivot_price

    if failed and not triggered:
        stage, name, confidence_note = SetupStage.FAILED, "vcp_failed", "disparó y volvió a caer, aviso"
    elif triggered:
        stage, name, confidence_note = SetupStage.TRIGGERED, "vcp_triggered", "ruptura confirmada"
    elif len(contractions) >= 3 and final_depth < VCP_FINAL_CONTRACTION_MAX_PCT:
        distance_atr = abs(pivot_price - price) / ctx.atr14 if ctx.atr14 else float("inf")
        if price <= pivot_price and distance_atr <= VCP_READY_MAX_DISTANCE_ATR:
            stage, name, confidence_note = SetupStage.READY, "vcp_ready", "listo, falta el disparo"
        else:
            return []
    else:
        stage = SetupStage.FORMING
        name = "vcp_forming"
        confidence_note = "formándose, última contracción aún ancha"

    # `contractions[-1][3]` es un índice dentro de la ventana recortada
    # (`close.iloc[-VCP_LOOKBACK_BARS:]`, reindexada desde 0) - `offset` es
    # cuántas barras se recortaron por delante, así que hay que sumarlo (no
    # restarlo) para volver a la posición real dentro de `ctx.close`.
    low2_idx_in_full_series = offset + contractions[-1][3]
    return [
        SetupMatch(
            family=SetupFamily.VCP,
            name=name,
            label_es=f"VCP ({len(contractions)} contracciones) - {confidence_note}",
            stage=stage,
            bars_in_stage=(len(ctx.close) - 1) - low2_idx_in_full_series,
            timeframe="daily",
            trigger_price=trigger_price if stage != SetupStage.FAILED else None,
            trigger_condition=(
                f"cierre por encima de {trigger_price:.2f} con volumen >= {VCP_TRIGGER_VOLUME_MULTIPLE}x"
            ),
            invalidation_price=invalidation_price,
            invalidation_condition=f"cierre por debajo de {invalidation_price:.2f} invalida el patrón",
            evidence=evidence,
            narrative_es=(
                f"VCP de {len(contractions)} contracciones ({', '.join(f'{d * 100:.0f}%' for d in depths)}), "
                f"con volumen decreciente en cada tramo - {confidence_note}."
            ),
            confidence=SetupConfidence.UNVALIDATED,
        )
    ]
