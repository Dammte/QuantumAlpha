"""Parte 4.1 del encargo: ruptura de nivel y caja de Darvas.

La ruptura de nivel reutiliza el motor de niveles casi por completo: un
nivel de resistencia con fuerza real (`LevelKind.PIVOT_RESISTANCE` con
`strength>=2`, o `RANGE_HIGH_20`/`HIGH_52W`, que no tienen `strength` -
Parte 4.1: "o ser un máximo de rango de al menos 20 sesiones") en estado
`LevelState.BROKEN_CONFIRMED` - ese estado, desde la Parte 5.1, YA exige su
propia confirmación de volumen (`BREAKOUT_CONFIRM_MIN_REL_VOLUME=1.2` sobre
una ventana de 21 sesiones) para pasar de `BREAKING` a `BROKEN_CONFIRMED`.
Este detector no re-deriva un segundo umbral de volumen independiente sobre
una ventana distinta - confía en que "confirmado" ya significa lo que
Parte 4.1 pide, con el mismo criterio de "no repitas ninguna pieza" que el
resto de la biblioteca.

La caja de Darvas sí es cómputo nuevo (el motor de niveles no tiene un
concepto de "rango comprimido reciente") - un rango de cierres de las
últimas N sesiones (20 a 60) con una anchura relativa por debajo de
`DARVAS_MAX_RANGE_PCT`. Se prueba de la ventana más larga a la más corta:
una caja que se sostiene más tiempo es una señal más fuerte que una recién
formada, así que la primera que encaja gana."""

from app.services import technical_analysis as ta
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

BREAKOUT_MAX_EXTENSION_ATR = 1.0
BREAKOUT_MIN_PIVOT_STRENGTH = 2
BREAKOUT_LEVEL_KINDS = (ta.LevelKind.PIVOT_RESISTANCE, ta.LevelKind.RANGE_HIGH_20, ta.LevelKind.HIGH_52W)

DARVAS_MAX_RANGE_PCT = 0.12
DARVAS_WINDOWS = (60, 50, 40, 30, 20)  # de más larga a más corta - una caja más duradera gana


def _matching_broken_level(levels: list[ta.Level]) -> ta.Level | None:
    for level in levels:
        if level.kind not in BREAKOUT_LEVEL_KINDS:
            continue
        if level.state != ta.LevelState.BROKEN_CONFIRMED:
            continue
        if level.side != "above":
            continue  # confirmado al alza - un BROKEN_CONFIRMED del lado bajista no es una ruptura
        if level.kind == ta.LevelKind.PIVOT_RESISTANCE and (
            level.strength is None or level.strength < BREAKOUT_MIN_PIVOT_STRENGTH
        ):
            continue
        if level.distance_atr > BREAKOUT_MAX_EXTENSION_ATR:
            continue  # ya muy extendido - llegar tarde, el stop ya no cabe bajo el nivel
        return level
    return None


def _check_level_breakout(ctx: SetupContext) -> SetupMatch | None:
    level = _matching_broken_level(ctx.levels)
    if level is None:
        return None

    evidence = {
        "level_kind": level.kind.value,
        "level_price": round(level.price, 4),
        "distance_atr": round(level.distance_atr, 3),
        "bars_since_break": level.bars_in_state,
    }
    return SetupMatch(
        family=SetupFamily.BREAKOUT,
        name="ruptura_de_nivel",
        label_es="Ruptura de nivel confirmada",
        stage=SetupStage.TRIGGERED,
        bars_in_stage=level.bars_in_state,
        timeframe="daily",
        trigger_price=level.price,
        trigger_condition=f"cierre por encima de {level.price:.2f} con volumen de confirmación",
        invalidation_price=level.price,
        invalidation_condition=f"cierre por debajo de {level.price:.2f} invalida la ruptura",
        evidence=evidence,
        narrative_es=(
            f"Ruptura confirmada de {level.kind.value} en {level.price:.2f}, a {level.distance_atr:.2f} "
            "ATR de distancia - todavía no extendida."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def _darvas_box(close) -> tuple[float, float, int] | None:
    """Los límites de la caja se miden EXCLUYENDO la barra de hoy a
    propósito: si se incluyera, un movimiento fuerte de hoy simplemente
    "ensancharía la caja" en vez de poder leerse como una ruptura de una
    caja que ya existía antes de hoy - `_check_darvas_box` es quien decide
    si el precio de hoy sigue dentro de estos límites o ya los rompió."""
    if len(close) < 2:
        return None
    prior_close = close.iloc[:-1]
    for window in DARVAS_WINDOWS:
        if len(prior_close) < window:
            continue
        recent = prior_close.iloc[-window:]
        box_high = float(recent.max())
        box_low = float(recent.min())
        if box_high <= 0:
            continue
        range_pct = (box_high - box_low) / box_high
        if range_pct < DARVAS_MAX_RANGE_PCT:
            return box_low, box_high, window
    return None


def _check_darvas_box(ctx: SetupContext) -> SetupMatch | None:
    box = _darvas_box(ctx.close)
    if box is None:
        return None
    box_low, box_high, window = box
    price = float(ctx.close.iloc[-1])
    if price > box_high or price < box_low:
        return None  # ya rompió la caja en cualquier sentido - eso es otro setup, no "caja vigente"

    evidence = {"box_high": round(box_high, 4), "box_low": round(box_low, 4), "window_sessions": window}
    return SetupMatch(
        family=SetupFamily.BREAKOUT,
        name="caja_de_darvas",
        label_es="Caja de Darvas",
        stage=SetupStage.READY,
        bars_in_stage=window,
        timeframe="daily",
        trigger_price=box_high,
        trigger_condition=f"cierre por encima del techo de la caja ({box_high:.2f})",
        invalidation_price=box_low,
        invalidation_condition=f"cierre por debajo del suelo de la caja ({box_low:.2f})",
        evidence=evidence,
        narrative_es=(
            f"Caja de Darvas de {window} sesiones entre {box_low:.2f} y {box_high:.2f} - consolidación "
            "estrecha, sin ambigüedad sobre dónde está el gatillo y dónde la anulación."
        ),
        confidence=SetupConfidence.UNVALIDATED,
    )


def detect(ctx: SetupContext) -> list[SetupMatch]:
    match = _check_level_breakout(ctx) or _check_darvas_box(ctx)
    return [match] if match is not None else []
