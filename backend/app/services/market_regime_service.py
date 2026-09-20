"""Auditoria del Radar, bloque G/10: "cabecera de régimen de mercado (una
sola línea): estado del índice de la región... amplitud... y su cambio en 5
sesiones. Con una regla operativa explícita: si el régimen es bajista, el
Radar sigue mostrando candidatos pero el tamaño de posición sugerido se
reduce a la mitad y el umbral del primario sube a 80. El contexto tiene que
tener consecuencia, no ser decorado" (literal).

Deliberadamente un módulo y un nombre DISTINTOS de
`market_context_service.assess_market_regime`/`MarketRegime` (el régimen
basado en VIX que ya alimenta el panel de contexto de mercado, una pregunta
de "estrés de volatilidad" completamente distinta) - reutilizar ese nombre
habría mezclado dos conceptos que la propia especificación pide por
separado. Este mide tendencia: el índice de la región frente a su propia
media de 30 semanas (`multi_timeframe.WEEKLY_STAGE_MA_WINDOW`), y la amplitud
del universo frente a la MISMA media, no SMA50/200 diarias.

Puro cálculo sobre datos ya calculados: `TickerDailyState.timeframe_strip`
(persistido por `daily_close.py` para la biblioteca de setups, nunca
recomputado aquí) da `weekly.price_vs_ma` por ticker - la amplitud es una
simple cuenta sobre eso. `index_above_weekly_ma30` es la única entrada que
no sale de una tabla precomputada (el propio endpoint la calcula con UNA
lectura en vivo del índice de referencia de la región, nunca del universo
completo) - se recibe ya calculada, este módulo no toca red ni pandas."""

from dataclasses import dataclass

from app.domain.models.ticker_daily_state import TickerDailyState

REGIME_BULLISH = "alcista"
REGIME_BEARISH = "bajista"
REGIME_UNKNOWN = "desconocido"

# Amplitud por debajo de la mitad del universo ya cuenta como bajista, aunque
# el índice en sí siga (por poco) sobre su propia media - la amplitud suele
# adelantarse al índice agregado (unos pocos pesos pesados pueden sostenerlo
# mientras la mayoría del universo ya se ha dado la vuelta).
BREADTH_BEARISH_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class MarketRegime:
    status: str  # "alcista" | "bajista" | "desconocido"
    index_above_weekly_ma30: bool | None
    breadth_pct: float | None  # % del universo analizado por encima de su propia MA30 semanal
    breadth_change_5d: float | None  # puntos (0-1) de diferencia vs. hace ~5 sesiones, None si no hay dato
    headline: str  # una sola línea, en español, lista para mostrar


def weekly_breadth_pct(states: list[TickerDailyState]) -> float | None:
    """Fracción de `states` cuyo `timeframe_strip.weekly.price_vs_ma` es
    "above" - `None` cuando ningún estado analizado trae esa lectura (sin
    universo, o filas anteriores a que `timeframe_strip` existiera)."""
    readings = [
        state.timeframe_strip["weekly"]["price_vs_ma"]
        for state in states
        if state.timeframe_strip is not None and state.timeframe_strip.get("weekly", {}).get("price_vs_ma")
    ]
    if not readings:
        return None
    above = sum(1 for r in readings if r == "above")
    return above / len(readings)


def assess_trend_regime(
    states: list[TickerDailyState],
    states_5d_ago: list[TickerDailyState] | None,
    index_above_weekly_ma30: bool | None,
) -> MarketRegime:
    """Bajista si el índice de la región está bajo su propia MA30 semanal, O
    la amplitud del universo cae por debajo de `BREADTH_BEARISH_THRESHOLD` -
    cualquiera de las dos basta, ninguna de las dos anula a la otra.
    `status == "desconocido"` (nunca se fabrica un veredicto) cuando falta
    cualquiera de las dos entradas - ni el índice en vivo ni la amplitud
    precomputada estuvieron disponibles."""
    breadth = weekly_breadth_pct(states)
    breadth_past = weekly_breadth_pct(states_5d_ago) if states_5d_ago else None
    breadth_change = breadth - breadth_past if breadth is not None and breadth_past is not None else None

    if index_above_weekly_ma30 is None or breadth is None:
        return MarketRegime(
            status=REGIME_UNKNOWN,
            index_above_weekly_ma30=index_above_weekly_ma30,
            breadth_pct=breadth,
            breadth_change_5d=breadth_change,
            headline="Régimen de mercado no disponible - falta historial suficiente para calcularlo hoy.",
        )

    is_bearish = not index_above_weekly_ma30 or breadth < BREADTH_BEARISH_THRESHOLD
    if is_bearish:
        reason = (
            "el índice está bajo su media de 30 semanas"
            if not index_above_weekly_ma30
            else f"solo el {breadth:.0%} del universo está sobre su propia media de 30 semanas"
        )
        headline = (
            f"Régimen bajista: {reason} - tamaño de posición reducido a la mitad y umbral de "
            "convicción más exigente."
        )
        status = REGIME_BEARISH
    else:
        headline = (
            f"Régimen alcista: índice sobre su media de 30 semanas, {breadth:.0%} del universo también."
        )
        status = REGIME_BULLISH

    return MarketRegime(
        status=status,
        index_above_weekly_ma30=index_above_weekly_ma30,
        breadth_pct=breadth,
        breadth_change_5d=breadth_change,
        headline=headline,
    )
