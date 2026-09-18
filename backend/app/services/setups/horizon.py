"""Auditoria del Radar, bloque D: horizonte corto vs medio plazo - "yo opero
posiciones de 2 a 10 sesiones, con entradas oportunistas... lo que a mí me
importa es el tiempo esperado hasta la resolución del setup", literal.

Deliberadamente NO derivado de `SetupMatch.timeframe` por sí solo como único
criterio de familia - un VCP diario que todavía necesita contracciones para
completarse no es "corto plazo" solo porque su indicador vive en temporalidad
diaria. El criterio real son dos señales objetivas, en este orden de
prioridad (nunca una lista de familias que mantener actualizada cada vez que
se añade un detector nuevo):

1. `timeframe == "weekly"` -> siempre `medium` ("requiere confirmación
   semanal", literal del encargo).
2. Si no, la distancia al gatillo en ATR decide: `stage == TRIGGERED` (ya
   disparado hoy, literal) o distancia <= `HORIZON_SHORT_MAX_DISTANCE_ATR`
   -> `short`; en cualquier otro caso (incluido no tener gatillo numérico
   todavía, típico de un FORMING sin nivel definido) -> `medium`, "necesita
   tiempo para madurar".

Esto reproduce la tabla del encargo como CONSECUENCIA, no como una lista
aparte: `vcp_forming` sin pivote definido cae en `medium` (sin distancia que
medir); `vcp_ready`/`vcp_triggered` cerca de su pivote
(`vcp.VCP_READY_MAX_DISTANCE_ATR=1.5`, casi siempre <= 1.0 en la práctica al
llegar a READY) cae en `short`; `stage1_base_forming` (semanal, sin gatillo
numérico) cae en `medium` por las dos señales a la vez; `stage2_confirmed`
(semanal, ya disparado) sigue en `medium` por vivir en temporalidad semanal
- coherente con "la señal es semanal" del propio encargo, aunque ya haya
confirmado.

No usa la duración de la base (`evidence["base_days"]`/`weeks_flat`...) como
tercera señal independiente, a propósito: cada familia guarda esa duración
con una clave distinta en `evidence` (dict libre, sin esquema común entre
detectores), así que parsearla de forma genérica aquí acoplaría este módulo
a los detalles internos de cada detector. La distancia al gatillo ya captura,
en la práctica, casi la misma información - una base larga sin resolver
todavía no tiene un gatillo cercano.

Paso posterior a la detección, nunca dentro de un detector - mismo patrón
que `arbitration.order_by_rank`/`context_modifiers.apply_context_modifiers`/
`setup_replay.apply_measured_confidence`, todos aplicados por
`daily_close.py` después de `registry.detect_all`."""

from dataclasses import replace

import pandas as pd

from app.services.setups.context import SetupContext
from app.services.setups.types import SetupMatch, SetupStage

HORIZON_SHORT_MAX_DISTANCE_ATR = 1.0  # literal del encargo: "menos de 1.0 ATR"
# Sesiones recientes sobre las que se mide el "recorrido medio diario" -
# un mes de calendario aproximado, suficiente para suavizar un par de días
# atípicos sin diluir un cambio de régimen de volatilidad reciente.
EXPECTED_SESSIONS_LOOKBACK = 20


def _distance_to_trigger_atr(match: SetupMatch, ctx: SetupContext) -> float | None:
    """`None` sin gatillo numérico (un FORMING sin nivel definido todavía)
    o sin ATR - nunca una aproximación fabricada."""
    if match.trigger_price is None or ctx.atr14 is None or ctx.atr14 <= 0 or len(ctx.close) == 0:
        return None
    price = float(ctx.close.iloc[-1])
    return abs(match.trigger_price - price) / ctx.atr14


def _expected_sessions_to_trigger(distance_atr: float | None, ctx: SetupContext) -> int | None:
    """Estimación simple y así etiquetada (Parte D, literal: "es aproximada
    y así debe etiquetarse en la UI") - distancia al gatillo en ATR dividida
    entre el recorrido medio diario reciente, también expresado en ATR (no
    en precio bruto, para que sea comparable entre tickers de precio y
    volatilidad distintos). `None` cuando no hay distancia que medir o el
    valor lleva `EXPECTED_SESSIONS_LOOKBACK` sesiones sin moverse en
    absoluto (recorrido medio 0 - dividir daría un resultado sin sentido,
    no infinito fabricado)."""
    if distance_atr is None or ctx.atr14 is None:
        return None
    if distance_atr <= 0:
        return 0
    recent_moves = ctx.close.diff().abs().iloc[-EXPECTED_SESSIONS_LOOKBACK:]
    avg_daily_move = float(recent_moves.mean()) if len(recent_moves) > 0 else float("nan")
    if pd.isna(avg_daily_move) or avg_daily_move <= 0:
        return None
    avg_daily_move_atr_fraction = avg_daily_move / ctx.atr14
    return max(1, round(distance_atr / avg_daily_move_atr_fraction))


def assign_horizon(matches: list[SetupMatch], ctx: SetupContext) -> list[SetupMatch]:
    """Asigna `horizon`/`expected_sessions_to_trigger` a cada coincidencia -
    ver el docstring del módulo para el criterio completo."""
    result = []
    for match in matches:
        distance_atr = _distance_to_trigger_atr(match, ctx)
        expected_sessions = _expected_sessions_to_trigger(distance_atr, ctx)

        if match.timeframe == "weekly":
            horizon = "medium"
        elif match.stage == SetupStage.TRIGGERED:
            horizon = "short"
        elif distance_atr is not None and distance_atr <= HORIZON_SHORT_MAX_DISTANCE_ATR:
            horizon = "short"
        else:
            horizon = "medium"

        result.append(replace(match, horizon=horizon, expected_sessions_to_trigger=expected_sessions))
    return result
