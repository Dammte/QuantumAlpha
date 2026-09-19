"""Auditoria del Radar, bloque E2: score compuesto 0-100, explicable por
componentes - "necesito saber por qué un valor está el primero, viendo el
desglose del score" (literal). Sustituye la tupla lexicográfica opaca que
`market.py::_radar_sort_key` usaba antes (con un `expectancy_rank`
hardcodeado a 0 que no ordenaba nada - peor que ausente, porque engañaba a
quien leyera el código).

Módulo puro (sin FastAPI/Pydantic/SQLAlchemy) - toma primitivos y devuelve
`RadarScoreBreakdown`, nunca un schema de `app/schemas/`. `market.py`
(la capa de API) es quien construye `RadarScoreInput` a partir de un
`RadarItemResponse` ya construido y adjunta el resultado a la respuesta.

Los cinco componentes y las dos penalizaciones están documentados con su
razonamiento en `app/core/trading_params.py`, junto a sus pesos - este
módulo solo implementa la aritmética, nunca redefine un peso propio."""

from dataclasses import dataclass
from datetime import date

from app.core import trading_params as tp

# Auditoria del Radar, bloque E2, literal: "próximas 10 sesiones" - se
# aproxima con días naturales, mismo criterio ya establecido por
# `trigger_performance_service.TAKEN_WINDOW_DAYS` para la misma clase de
# pregunta ("¿cuántas sesiones de aquí a X"). Un valor MENOR que 10 sesiones
# reales de calendario (en vez de ~14, que sería la conversión exacta con
# fines de semana) para que la advertencia de earnings avise pronto, nunca
# tarde - más conservador, no menos.
_EARNINGS_WINDOW_CALENDAR_DAYS = tp.RADAR_SCORE_EARNINGS_WITHIN_SESSIONS

# `entry_type` (de `trade_geometry.EntryType`, ya expuesto como `str` en
# `TradeGeometryResponse`) que corresponde a un nivel de precio real
# (ruptura/soporte) en vez de una media móvil - ver `_stop_cascade` en
# `trade_geometry.py`. Todo `entry_geometry` viable ya tiene UN anclaje de
# los dos grupos (nunca "sin anclaje" - `_stop_cascade` rechaza la geometría
# entera si no encuentra ninguno), así que este componente distingue calidad
# de anclaje, no presencia/ausencia.
_REAL_LEVEL_ENTRY_TYPES = frozenset({"breakout", "pullback_support"})


@dataclass(frozen=True, slots=True)
class RadarScoreInput:
    """Todo lo que `compute_radar_score` necesita, ya resuelto por quien lo
    llama - `kw_only` de facto (todos los campos con nombre al construirlo)
    porque son más de media docena y una llamada posicional sería frágil."""

    grade: str | None  # "A" | "B" | "C" | None
    setup_confidence: str | None  # "measured" | "thin" | "unvalidated" | None
    rs_rating: int | None
    sector_rs_percentile: int | None
    distance_atr: float | None  # distancia al gatillo del setup líder, en ATR
    risk_reward_net: float | None
    entry_type: str | None  # `trade_geometry.EntryType.value` de la geometría propuesta
    relative_volume: float | None
    setup_stage: str | None  # "forming" | "ready" | "triggered" | "failed" | None
    horizon: str | None  # "short" | "medium" | None
    next_earnings_date: date | None
    as_of: date
    atr_pct: float | None
    atr_pct_p90_in_universe: float | None  # percentil 90 de atr_pct entre los candidatos de hoy


@dataclass(frozen=True, slots=True)
class RadarScoreBreakdown:
    """Cada componente ya multiplicado por su peso (0 al peso máximo de
    `trading_params.RADAR_SCORE_WEIGHT_*`) y cada penalización ya en
    negativo - `total` es la suma de todo, recortada a `[0, 100]`. Se expone
    tal cual en la respuesta de la API (bloque E2: "el score y el desglose
    por componente viajan en la respuesta")."""

    total: float
    setup_quality: float
    relative_strength: float
    trigger_proximity: float
    geometry_quality: float
    volume_confirmation: float
    earnings_penalty: float  # 0, o negativo
    high_atr_penalty: float  # 0, o negativo


def _setup_quality(grade: str | None, confidence: str | None) -> float:
    grade_points = tp.RADAR_SCORE_GRADE_POINTS.get(grade or "", 0.0)
    confidence_mult = tp.RADAR_SCORE_CONFIDENCE_MULTIPLIER.get(confidence or "unvalidated", 0.5)
    return grade_points * confidence_mult * tp.RADAR_SCORE_WEIGHT_SETUP_QUALITY


def _relative_strength(rs_rating: int | None, sector_rs_percentile: int | None) -> float:
    ticker_component = (rs_rating or 0) / 100 * tp.RADAR_SCORE_TICKER_RS_WEIGHT
    sector_component = (sector_rs_percentile or 0) / 100 * tp.RADAR_SCORE_SECTOR_RS_WEIGHT
    return (ticker_component + sector_component) * tp.RADAR_SCORE_WEIGHT_RELATIVE_STRENGTH


def _trigger_proximity(distance_atr: float | None) -> float:
    if distance_atr is None:
        return 0.0
    fraction = max(0.0, 1.0 - distance_atr / tp.RADAR_SCORE_PROXIMITY_ATR_SCALE)
    return fraction * tp.RADAR_SCORE_WEIGHT_TRIGGER_PROXIMITY


def _geometry_quality(risk_reward_net: float | None, entry_type: str | None) -> float:
    if risk_reward_net is None:
        rr_fraction = 0.0
    else:
        span = tp.RADAR_SCORE_RISK_REWARD_CEILING - tp.MIN_RISK_REWARD_NET
        rr_fraction = max(0.0, min(1.0, (risk_reward_net - tp.MIN_RISK_REWARD_NET) / span)) if span > 0 else 0.0
    if entry_type in _REAL_LEVEL_ENTRY_TYPES:
        anchor_fraction = tp.RADAR_SCORE_REAL_LEVEL_ANCHOR_SCORE
    elif entry_type is not None:
        anchor_fraction = tp.RADAR_SCORE_MA_ANCHOR_SCORE
    else:
        anchor_fraction = 0.0
    fraction = (
        rr_fraction * tp.RADAR_SCORE_RISK_REWARD_SUBWEIGHT + anchor_fraction * tp.RADAR_SCORE_ANCHOR_SUBWEIGHT
    )
    return fraction * tp.RADAR_SCORE_WEIGHT_GEOMETRY_QUALITY


def _volume_confirmation(relative_volume: float | None, setup_stage: str | None) -> float:
    # Sin dato, neutral - ni premia ni penaliza (a diferencia de la
    # confianza `unvalidated`, aquí no hay una lectura "conservadora" obvia
    # que fabricar sin el número real).
    if relative_volume is None:
        fraction = 0.5
    elif setup_stage == "triggered":
        floor, ceiling = tp.RADAR_SCORE_VOLUME_TRIGGERED_FLOOR, tp.RADAR_SCORE_VOLUME_TRIGGERED_CEILING
        fraction = max(0.0, min(1.0, (relative_volume - floor) / (ceiling - floor)))
    else:
        # FORMING/READY: menos volumen (contracción) puntúa más alto.
        floor, ceiling = tp.RADAR_SCORE_VOLUME_FORMING_FLOOR, tp.RADAR_SCORE_VOLUME_FORMING_CEILING
        fraction = max(0.0, min(1.0, (ceiling - relative_volume) / (ceiling - floor)))
    return fraction * tp.RADAR_SCORE_WEIGHT_VOLUME_CONFIRMATION


def _earnings_penalty(next_earnings_date: date | None, as_of: date, horizon: str | None) -> float:
    # "En medio plazo, solo aviso" (literal) - la penalización numérica es
    # exclusiva del horizonte corto, donde entrar antes de resultados
    # "no es análisis técnico, es una moneda al aire".
    if horizon != "short" or next_earnings_date is None:
        return 0.0
    days_until = (next_earnings_date - as_of).days
    if 0 <= days_until <= _EARNINGS_WINDOW_CALENDAR_DAYS:
        return -tp.RADAR_SCORE_EARNINGS_PENALTY_SHORT
    return 0.0


def _high_atr_penalty(atr_pct: float | None, atr_pct_p90: float | None) -> float:
    if atr_pct is None or atr_pct_p90 is None or atr_pct <= atr_pct_p90:
        return 0.0
    return -tp.RADAR_SCORE_HIGH_ATR_PENALTY


def compute_radar_score(inp: RadarScoreInput) -> RadarScoreBreakdown:
    setup_quality = _setup_quality(inp.grade, inp.setup_confidence)
    relative_strength = _relative_strength(inp.rs_rating, inp.sector_rs_percentile)
    trigger_proximity = _trigger_proximity(inp.distance_atr)
    geometry_quality = _geometry_quality(inp.risk_reward_net, inp.entry_type)
    volume_confirmation = _volume_confirmation(inp.relative_volume, inp.setup_stage)
    earnings_penalty = _earnings_penalty(inp.next_earnings_date, inp.as_of, inp.horizon)
    high_atr_penalty = _high_atr_penalty(inp.atr_pct, inp.atr_pct_p90_in_universe)

    raw_total = (
        setup_quality
        + relative_strength
        + trigger_proximity
        + geometry_quality
        + volume_confirmation
        + earnings_penalty
        + high_atr_penalty
    )
    total = max(0.0, min(100.0, raw_total))

    return RadarScoreBreakdown(
        total=total,
        setup_quality=setup_quality,
        relative_strength=relative_strength,
        trigger_proximity=trigger_proximity,
        geometry_quality=geometry_quality,
        volume_confirmation=volume_confirmation,
        earnings_penalty=earnings_penalty,
        high_atr_penalty=high_atr_penalty,
    )
