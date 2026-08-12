"""Classifies *when* in a setup's life you'd be entering right now, separately
from *whether* the setup is good at all (that's still the recommendation
engine's job - see `recommendation_engine.py`).

The motivating problem: a name can pass every trend/RS/Minervini criterion and
still be a bad trade to enter today, because the easy, low-risk part of the
move already happened before it was even flagged - "el setup era bueno hace
dos semanas" is a real and common way a technically-sound signal turns into a
losing entry. This is a *timing* read layered on top of the recommendation,
not a replacement for it: it never argues a ticker is good or bad, only how
much of its current move looks still-ahead versus already-behind.

Deliberately reuses two signals the recommendation engine already computes for
other purposes rather than inventing a new indicator:

- `atr_multiple` - how many Average True Ranges price sits above its own
  50-day SMA (see `technical_analysis.atr_multiple_from_sma`) - already used
  by `recommendation_engine.py`'s "Extensión parabólica" factor (penalized
  outright above 4.0). A volatility-normalized "how far has this run from its
  own recent base" gauge: the same 2-ATR move means something very different
  for a sleepy utility than for a momentum semiconductor name, and this
  already accounts for that.
- `nearest_support` - see `technical_analysis.support_resistance_levels` -
  already used by the same engine's "Rebotando en un soporte cercano" factor.
  Price sitting right on a support in an established uptrend is the closest
  this system gets to a textbook low-risk pullback entry (Minervini/O'Neil's
  "buy point" idea, and Weinstein's Stage 2 pullback-to-support entries).

Four bands, ordered from best to worst risk/reward for a *new* entry today:

- OPTIMAL: right at (or barely off) its base/support in an intact uptrend -
  the setup's own natural low-risk entry point.
- VALID: has already moved up some, but still within a reasonable distance of
  its base - a later, somewhat worse-odds entry than optimal, not a bad one.
- LATE: meaningfully extended (2.5-4 ATRs above its 50-day SMA) - chasing the
  move at this point carries real risk of entering right before a pause or
  pullback, even if the underlying trend keeps going afterward.
- EXTENDED: beyond 4 ATRs - the same "parabolic" zone the recommendation
  engine already penalizes; historically the highest-risk point to open a new
  position, reversion-prone regardless of trend quality.
"""

from dataclasses import dataclass

from app.services import technical_analysis as ta

OPTIMAL = "optimal"
VALID = "valid"
LATE = "late"
EXTENDED = "extended"

_LABELS: dict[str, str] = {
    OPTIMAL: "Momento óptimo de entrada",
    VALID: "Entrada aún válida",
    LATE: "Entrada tardía",
    EXTENDED: "Muy extendido - alto riesgo de entrada",
}

_DESCRIPTIONS: dict[str, str] = {
    OPTIMAL: (
        "El precio está en o cerca de su base/soporte, con la tendencia intacta - el punto de entrada con "
        "mejor relación riesgo/recompensa que ofrece este setup ahora mismo."
    ),
    VALID: (
        "El precio ya se ha alejado algo de su base, pero sigue dentro de un rango razonable para entrar - "
        "todavía queda recorrido, aunque ya no es el punto de entrada más favorable."
    ),
    LATE: (
        "El precio se ha extendido de forma notable desde su base (entre 2,5 y 4 veces su rango medio, "
        "ATR). Entrar ahora implica perseguir el movimiento: puede seguir subiendo, pero el riesgo de una "
        "pausa o retroceso antes de que continúe es mayor que en una entrada temprana."
    ),
    EXTENDED: (
        "El precio está muy extendido de su media (más de 4 veces su ATR sobre la SMA50) - la misma zona "
        "que ya penaliza el motor de recomendación como \"extensión parabólica\". Históricamente es el punto "
        "de mayor riesgo para abrir una posición nueva, incluso si la tendencia de fondo sigue siendo buena."
    ),
}

# Matches recommendation_engine.py's own thresholds (SUPPORT_PROXIMITY, and the
# >4 "parabolic" cutoff) so this reads as the same underlying facts, viewed
# through a timing lens, not a second, disagreeing opinion.
EXTENDED_ATR_THRESHOLD = 4.0
LATE_ATR_THRESHOLD = 2.5
OPTIMAL_ATR_THRESHOLD = 1.2
SUPPORT_PROXIMITY = 0.03


@dataclass(frozen=True, slots=True)
class EntryTiming:
    status: str  # "optimal" | "valid" | "late" | "extended"
    label: str
    description: str
    atr_multiple: float


def assess_entry_timing(
    atr_multiple: float | None,
    nearest_support: ta.PriceLevel | None,
    trend: ta.TrendState,
) -> EntryTiming | None:
    """None when `atr_multiple` itself is unavailable (not enough history for
    an ATR/SMA50 read) - same graceful-degradation posture as every other
    optional signal in this codebase, rather than guessing."""
    if atr_multiple is None:
        return None

    near_support = nearest_support is not None and abs(nearest_support.distance_pct) <= SUPPORT_PROXIMITY

    if atr_multiple > EXTENDED_ATR_THRESHOLD:
        status = EXTENDED
    elif atr_multiple > LATE_ATR_THRESHOLD:
        status = LATE
    elif atr_multiple <= OPTIMAL_ATR_THRESHOLD or (near_support and trend == ta.TrendState.UPTREND):
        status = OPTIMAL
    else:
        status = VALID

    return EntryTiming(
        status=status, label=_LABELS[status], description=_DESCRIPTIONS[status], atr_multiple=atr_multiple
    )
