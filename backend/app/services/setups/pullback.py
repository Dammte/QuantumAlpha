"""Parte 4.2 del encargo: retroceso en tendencia - "el setup de mejor
geometría de todos, porque el stop queda muy cerca". Reutiliza el motor de
niveles real (`technical_analysis.detect_levels`, Parte 5.1) para "a menos
de 0,5 ATR de la EMA21/55 o de un pivote de soporte": eso es exactamente
`LevelState.TESTING` (definido como "< 0,5 ATR" en el propio módulo), así
que este detector no mide ninguna distancia por su cuenta - lee el estado
que `ctx.levels` ya trae.

**Simplificación deliberada, documentada**: "el retroceso no supera el 50%
del último impulso (del último pivote mínimo al último máximo)" pide un
detector de pivotes indexado en el tiempo - `technical_analysis._fractal_pivots`
(el que ya existe) solo devuelve precios, no su posición, así que no basta
para "el último mínimo antes del último máximo" tal cual. `_last_impulse`
usa en su lugar el máximo de una ventana reciente y el mínimo de `low`
ANTES de ese máximo dentro de la misma ventana - una aproximación al mismo
concepto sin necesitar un detector de pivotes nuevo, ver quant_methodology.md
§28.5 para el porqué completo."""

from app.services import multi_timeframe as mtf
from app.services import technical_analysis as ta
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage

# Corrección propia (§28.1): el texto original pedía "media de 3 sesiones <
# 0,9x media de 20" - 0,9 apenas filtra nada, la mayoría de sesiones caen
# ahí por varianza normal del volumen. 0,75 es el mismo umbral corregido
# que ya se documentó antes de escribir el primer detector.
PULLBACK_VOLUME_DRYUP_RATIO = 0.75
PULLBACK_RSI_MIN = 40.0
PULLBACK_RSI_MAX = 55.0
PULLBACK_MAX_RETRACEMENT_PCT = 0.5
PULLBACK_IMPULSE_LOOKBACK_BARS = 60
PULLBACK_INVALIDATION_ATR = 0.4
PULLBACK_MIN_PIVOT_STRENGTH = 2
PULLBACK_LEVEL_KINDS = (ta.LevelKind.EMA21, ta.LevelKind.EMA55, ta.LevelKind.PIVOT_SUPPORT)


def _matching_level(levels: list[ta.Level]) -> ta.Level | None:
    for level in levels:
        if level.kind not in PULLBACK_LEVEL_KINDS:
            continue
        if level.state != ta.LevelState.TESTING:
            continue
        if level.side != "above":
            continue  # retrocediendo HACIA el nivel desde arriba, no ya roto por debajo de él
        if level.kind == ta.LevelKind.PIVOT_SUPPORT and (
            level.strength is None or level.strength < PULLBACK_MIN_PIVOT_STRENGTH
        ):
            continue
        return level
    return None


def _volume_drying_up(volume) -> bool:
    avg3 = volume.rolling(3).mean()
    avg20 = volume.rolling(20).mean()
    ratio = (avg3 / avg20).dropna()
    if ratio.empty:
        return False
    return bool(ratio.iloc[-1] < PULLBACK_VOLUME_DRYUP_RATIO)


def _last_impulse(high, low, lookback: int = PULLBACK_IMPULSE_LOOKBACK_BARS) -> tuple[float, float] | None:
    if high.empty:
        return None
    window_high = high.iloc[-lookback:]
    window_low = low.iloc[-lookback:]
    high_idx = window_high.idxmax()
    impulse_high = float(window_high.loc[high_idx])
    low_before_high = window_low.loc[:high_idx]
    if low_before_high.empty:
        return None
    return float(low_before_high.min()), impulse_high


def detect(ctx: SetupContext) -> list[SetupMatch]:
    if ctx.trend != ta.TrendState.UPTREND:
        return []
    if mtf.timeframe_bias(ctx.multi_timeframe.weekly) == "bearish":
        return []

    level = _matching_level(ctx.levels)
    if level is None:
        return []

    if not _volume_drying_up(ctx.volume):
        return []

    if ctx.rsi14 is None or not (PULLBACK_RSI_MIN <= ctx.rsi14 <= PULLBACK_RSI_MAX):
        return []

    impulse = _last_impulse(ctx.high, ctx.low)
    if impulse is None:
        return []
    impulse_low, impulse_high = impulse
    if impulse_high <= impulse_low:
        return []
    price = float(ctx.close.iloc[-1])
    retracement_pct = (impulse_high - price) / (impulse_high - impulse_low)
    if retracement_pct > PULLBACK_MAX_RETRACEMENT_PCT:
        return []

    trigger_price = float(ctx.high.iloc[-2]) if len(ctx.high) >= 2 else None
    invalidation_price = level.price - PULLBACK_INVALIDATION_ATR * ctx.atr14 if ctx.atr14 is not None else None

    evidence: dict[str, float | int | str] = {
        "level_kind": level.kind.value,
        "distance_atr": round(level.distance_atr, 3),
        "retracement_pct": round(retracement_pct, 3),
        "rsi14": round(ctx.rsi14, 1),
    }
    return [
        SetupMatch(
            family=SetupFamily.PULLBACK,
            name="pullback_a_media_o_soporte",
            label_es="Retroceso en tendencia",
            stage=SetupStage.READY,
            bars_in_stage=level.bars_in_state,
            timeframe="daily",
            trigger_price=trigger_price,
            trigger_condition=(
                f"cierre por encima del máximo de la sesión anterior, o recuperación de {level.kind.value}"
            ),
            invalidation_price=invalidation_price,
            invalidation_condition=(
                f"cierre por debajo de {invalidation_price:.2f} (nivel menos {PULLBACK_INVALIDATION_ATR} ATR)"
                if invalidation_price is not None
                else ""
            ),
            evidence=evidence,
            narrative_es=(
                f"Retroceso de tendencia alcista hasta {level.kind.value}, a {level.distance_atr:.2f} ATR. "
                f"El volumen se seca y el RSI ({ctx.rsi14:.0f}) se enfría sin romperse. Retroceso del "
                f"{retracement_pct * 100:.0f}% del último impulso - stop cerca, geometría favorable."
            ),
            confidence=SetupConfidence.UNVALIDATED,
        )
    ]
