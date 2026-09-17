"""Tipo común de la biblioteca de setups (Radar, Fase 1): el vocabulario que
todo detector de `setups/*.py` comparte, sin ninguna lógica de detección
todavía - eso vive en un módulo por familia (`stage_transition.py`, `vcp.py`,
etc.), nunca aquí.

Regla de diseño no negociable (ver `setups/__init__.py`): un `SetupMatch` es
una afirmación local de un solo detector sobre un solo ticker en un momento
dado - nunca lleva una puntuación que se sume con la de otro. Comparar/
elegir entre varios `SetupMatch` de un mismo ticker es responsabilidad de
`setups/arbitration.py` (fase posterior), no de este módulo ni del propio
detector.
"""

from dataclasses import dataclass
from enum import Enum


class SetupFamily(str, Enum):
    STAGE_TRANSITION = "stage_transition"
    VCP = "vcp"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    MA_CROSS = "ma_cross"
    CHANNEL = "channel"
    CLASSIC_PATTERN = "classic_pattern"


class SetupStage(str, Enum):
    """En qué punto de su formación está el setup - lo que distingue "empieza
    a mirarlo" de "actúa" (ver `setups/__init__.py`)."""

    FORMING = "forming"  # se está construyendo, no hay nada que hacer
    READY = "ready"  # la estructura está completa, falta el disparo
    TRIGGERED = "triggered"  # el disparo ocurrió y está confirmado
    FAILED = "failed"  # disparó y falló; se conserva N sesiones como aviso


class SetupConfidence(str, Enum):
    """Qué tan medido está este setup en concreto - ver `setup_replay.py`
    (fase posterior) y la Parte 10 del encargo original. Nunca se fabrica:
    un detector nuevo siempre empieza en `UNVALIDATED`, nunca en `MEASURED`."""

    MEASURED = "measured"  # tiene muestra histórica suficiente (>= MIN_SAMPLE_FOR_STATS)
    THIN = "thin"  # tiene muestra, pero por debajo del mínimo
    UNVALIDATED = "unvalidated"  # detector nuevo, sin replay todavía


@dataclass(frozen=True, slots=True)
class SetupMatch:
    """Una coincidencia de un detector sobre un ticker, en un momento dado.
    Campos tal como los pide el encargo original - ver `setups/__init__.py`
    para las reglas de uso (ningún campo aquí es una puntuación)."""

    family: SetupFamily
    name: str  # "vcp_3_contracciones", "stage1_to_2_rs_girando" - identifica el subestado exacto
    label_es: str  # texto para la interfaz
    stage: SetupStage
    bars_in_stage: int
    timeframe: str  # "daily" | "weekly"

    trigger_price: float | None
    trigger_condition: str  # "cierre > 118,40 con volumen >= 1,5x"
    invalidation_price: float | None
    invalidation_condition: str  # "cierre < 108,90 rompe la base"

    evidence: dict[str, float | int | str]  # los números que lo justifican, para verificar a ojo
    narrative_es: str  # una frase determinista (plantilla + números, nunca Gemini) del estado actual

    confidence: SetupConfidence


_SETUP_MATCH_PLAIN_FIELDS = (
    "name",
    "label_es",
    "bars_in_stage",
    "timeframe",
    "trigger_price",
    "trigger_condition",
    "invalidation_price",
    "invalidation_condition",
    "evidence",
    "narrative_es",
)


def setup_match_to_dict(match: SetupMatch) -> dict:
    """Plain JSON-safe read of a `SetupMatch` - mismo patrón que
    `trade_geometry.geometry_to_dict`, para que `scripts/daily_close.py`
    pueda persistir `TickerDailyState.setups` sin un mapeador propio de
    esquema. `family`/`stage`/`confidence` pasan a su `.value` explícito -
    aunque los tres son ya subclases de `str`, esta biblioteca no confía en
    ese detalle de implementación para lo que se guarda en una columna JSON."""
    data = {field: getattr(match, field) for field in _SETUP_MATCH_PLAIN_FIELDS}
    data["family"] = match.family.value
    data["stage"] = match.stage.value
    data["confidence"] = match.confidence.value
    return data


def setup_match_from_dict(data: dict) -> SetupMatch:
    """Inversa de `setup_match_to_dict` - reconstruye un `SetupMatch` real,
    no solo su forma de visualización, por si una fase futura necesita
    reprocesarlo (p. ej. el replay de `setup_replay.py`)."""
    fields = {field: data[field] for field in _SETUP_MATCH_PLAIN_FIELDS}
    return SetupMatch(
        family=SetupFamily(data["family"]),
        stage=SetupStage(data["stage"]),
        confidence=SetupConfidence(data["confidence"]),
        **fields,
    )
