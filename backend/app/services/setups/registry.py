"""El registro de detectores - un único punto de entrada, `detect_all`, que
`daily_close.py` llama una vez por ticker (§28.2 en adelante).

Cada familia se añade explícitamente a `SETUP_DETECTORS` cuando su propio
detector, sus tests y su constante de versión existen - nunca antes. Un
detector que todavía no está en esta lista simplemente no corre, no es un
caso especial que `detect_all` tenga que conocer.

Aislamiento por detector, no solo por ticker: `PortfolioRiskService` ya
documenta (ver su docstring) que un fallo de una posición no puede tumbar el
resto de la cartera - la misma disciplina aplica aquí un nivel más abajo: un
detector con un borde no cubierto (p. ej. VCP con menos barras de las que
espera) no debe vaciar el resultado de `stage_transition`/`ma_cross` del
mismo ticker."""

import logging
from collections.abc import Callable

from app.services.setups import breakout, channel, ma_cross, pullback, stage_transition
from app.services.setups.context import SetupContext
from app.services.setups.types import SetupMatch

logger = logging.getLogger(__name__)

SetupDetector = Callable[[SetupContext], list[SetupMatch]]

SETUP_DETECTORS: list[SetupDetector] = [
    stage_transition.detect,  # Parte 2 - transición Weinstein etapa 1 -> 2
    ma_cross.detect,  # Parte 4.3 - cruce rápido EMA21/55 al alza
    pullback.detect,  # Parte 4.2 - retroceso en tendencia
    breakout.detect,  # Parte 4.1 - ruptura de nivel / caja de Darvas
    channel.detect,  # Parte 4.4 - canales por regresión lineal
]


def detect_all(ctx: SetupContext) -> list[SetupMatch]:
    """Ejecuta todos los detectores registrados sobre un ticker y devuelve
    todas las coincidencias, sin ordenar ni elegir entre ellas - eso es
    trabajo de `setups/arbitration.py` (fase posterior), no de este
    registro."""
    matches: list[SetupMatch] = []
    for detector in SETUP_DETECTORS:
        try:
            matches.extend(detector(ctx))
        except Exception:
            logger.exception(
                "Setup detector %s failed for %s - skipping only this detector, not the ticker",
                getattr(detector, "__module__", detector),
                ctx.ticker,
            )
    return matches
