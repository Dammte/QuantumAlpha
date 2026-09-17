"""Parte 0/6 del encargo: la regla de diseño no negociable de toda la
biblioteca, en un único módulo. "Ningún setup suma puntos a otro" - un
ticker que cumple varios setups a la vez no obtiene "más nota" por ello, se
queda con el mejor de todos, y el resto se muestra como contexto. Este es
el único sitio donde se compara/elige entre varios `SetupMatch` de un mismo
ticker - `registry.detect_all` deliberadamente no ordena ni elige nada (ver
su propio docstring), y ningún detector individual sabe que otras familias
existen.

Los moduladores de contexto (Parte 6: squeeze, pocket pivot, secado de
volumen, contracción de ATR, ruptura fallida reciente, coincidencia de
setups) suben como máximo un escalón de grado EN TOTAL - esa parte vive en
`setups/context.py` (fase posterior, todavía no construida) y se aplicará
aquí mismo una vez exista; por ahora este módulo solo resuelve "cuál es el
mejor setup de los que coinciden", el criterio más básico y ya con
consumidores reales (los cinco detectores hasta ahora)."""

from app.services.setups.types import SetupMatch, SetupStage

# Orden de más a menos avanzado - el mismo criterio que la Parte 9.1 ya usa
# como primera clave de ordenación del propio Radar ("SetupStage: TRIGGERED
# -> READY -> FORMING. Lo que ya disparó primero."), reutilizado aquí para
# decidir cuál es el setup "titular" de un ticker cuando varias familias
# coinciden a la vez.
_STAGE_RANK = {SetupStage.TRIGGERED: 0, SetupStage.READY: 1, SetupStage.FORMING: 2, SetupStage.FAILED: 3}


def select_best(matches: list[SetupMatch]) -> SetupMatch | None:
    """El setup titular entre los que coinciden para un mismo ticker - `None`
    si `matches` está vacío. Empates de etapa se resuelven por el orden en
    que llegaron los propios matches (el orden de `SETUP_DETECTORS` en
    `registry.py`) - una regla de desempate estable y documentada, no un
    criterio inventado sin respaldo: el encargo no da un criterio explícito
    para comparar setups de familias distintas en la misma etapa, así que
    esto es lo mínimo defendible en vez de un orden arbitrario no
    determinista."""
    if not matches:
        return None
    return min(matches, key=lambda m: _STAGE_RANK.get(m.stage, len(_STAGE_RANK)))


def order_by_rank(matches: list[SetupMatch]) -> list[SetupMatch]:
    """Los mismos `matches`, reordenados con el titular primero (índice 0) y
    el resto como contexto detrás - la lista que `daily_close.py` persiste
    en `TickerDailyState.setups`. Nunca elimina ninguno: "los otros dos se
    muestran como contexto" (Parte 0), no se descartan."""
    if len(matches) <= 1:
        return list(matches)
    best = select_best(matches)
    rest = [m for m in matches if m is not best]
    return [best, *rest]
