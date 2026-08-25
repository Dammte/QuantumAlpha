"""Classic business-cycle sector rotation: cross-references which of the 11
sectors are actually leading right now (their RS-rank percentile, already
computed by `market_screener_service.get_sector_performance`) against the
well-established sector-leadership pattern each phase of the economic cycle
tends to produce (the Fidelity / Sam Stovall "stock market cycle" model).

This is deliberately price-driven, not a news/causal read: whatever the
reason a sector is leading - a war pushing energy higher, an earnings season,
a rate cut, anything - if it shows up as genuine relative-strength
outperformance in the data, this surfaces it without needing to read or
interpret any news. No attempt is made to explain *why* a sector is leading,
only to name what's leading and what that pattern has historically meant.
"""

from dataclasses import dataclass

from app.domain.models.ticker_snapshot import SectorPerformance

DEFAULT_TOP_N = 3

DEFENSIVE_SECTORS = {"Consumo defensivo", "Utilities", "Salud"}

# Each phase's canonical leadership set (Fidelity/Stovall-style business-cycle
# rotation). Overlap between these and the actual current top-N leaders picks
# the closest-matching phase - not a forecast, just a pattern match.
CYCLE_PHASE_LEADERS: dict[str, set[str]] = {
    "recuperación temprana": {"Consumo discrecional", "Financiero", "Inmobiliario", "Industrial"},
    "expansión media": {"Tecnología", "Comunicación", "Industrial"},
    "expansión tardía": {"Energía", "Materiales", "Consumo defensivo"},
    "contracción / recesión": {"Consumo defensivo", "Utilities", "Salud"},
}

CYCLE_PHASE_DESCRIPTIONS: dict[str, str] = {
    "recuperación temprana": (
        "Los sectores cíclicos sensibles a tasas (financiero, inmobiliario, consumo discrecional) están "
        "liderando - patrón típico de una recuperación temprana del ciclo económico."
    ),
    "expansión media": (
        "Tecnología, comunicación e industriales liderando - patrón típico de una expansión ya en marcha."
    ),
    "expansión tardía": (
        "Energía y materiales liderando, con el consumo defensivo ganando terreno - patrón típico de una "
        "expansión tardía, con presión inflacionaria y el mercado empezando a rotar hacia protección."
    ),
    "contracción / recesión": (
        "Los sectores defensivos (consumo básico, utilities, salud) están liderando mientras los cíclicos "
        "se quedan atrás - patrón típico de una fase de contracción o recesión: el mercado está rotando "
        "hacia protección de capital."
    ),
}


@dataclass(frozen=True, slots=True)
class SectorRotationSummary:
    leaders: list[str]
    laggards: list[str]
    cycle_phase: str | None
    cycle_confidence: float
    cycle_description: str | None
    defensive_leadership: bool
    warning: str | None


def assess_sector_rotation(
    sector_performance: list[SectorPerformance], top_n: int = DEFAULT_TOP_N
) -> SectorRotationSummary | None:
    ranked = sorted(
        (s for s in sector_performance if s.rs_rank is not None), key=lambda s: s.rs_rank, reverse=True
    )
    if not ranked:
        return None

    leaders = [s.sector for s in ranked[:top_n]]
    # Tercera auditoría, Bloque A-7: with fewer than 2*top_n sectors actually
    # ranked (a thin universe, or several sectors missing performance data),
    # a plain `ranked[-top_n:]` re-included sectors already claimed as
    # leaders - the same sector could show up as both "leading" and
    # "lagging" at once. Starting the laggard slice no earlier than `top_n`
    # means it shrinks (fewer laggards reported) instead of overlapping.
    laggard_start = max(top_n, len(ranked) - top_n)
    laggards = [s.sector for s in ranked[laggard_start:]][::-1]
    leader_set = set(leaders)

    # Tercera auditoría, Bloque A-7: `if overlap > best_overlap` alone let a
    # tie always resolve to whichever phase happens to iterate first
    # ("recuperación temprana" - which also has 4 sectors in its leadership
    # set against the other three phases' 3, a second, compounding bias
    # toward it on any near-tie). A genuine tie between phases means the
    # data doesn't actually distinguish between them - report no phase
    # rather than pick one by insertion order.
    best_overlap = -1
    tied_phases: list[str] = []
    for phase, phase_sectors in CYCLE_PHASE_LEADERS.items():
        overlap = len(leader_set & phase_sectors)
        if overlap > best_overlap:
            best_overlap, tied_phases = overlap, [phase]
        elif overlap == best_overlap:
            tied_phases.append(phase)
    best_phase = tied_phases[0] if len(tied_phases) == 1 else None
    confidence = best_overlap / top_n if top_n and best_overlap > 0 and best_phase is not None else 0.0

    defensive_leadership = len(leader_set & DEFENSIVE_SECTORS) >= 2
    warning = None
    if defensive_leadership:
        warning = (
            "La mayoría de los sectores líderes ahora mismo son defensivos (consumo básico, utilities, "
            "salud) - esto suele ser una señal de cautela del mercado en su conjunto, no solo de esos "
            "sectores puntuales."
        )

    return SectorRotationSummary(
        leaders=leaders,
        laggards=laggards,
        cycle_phase=best_phase if confidence > 0 else None,
        cycle_confidence=confidence,
        cycle_description=CYCLE_PHASE_DESCRIPTIONS.get(best_phase) if confidence > 0 and best_phase else None,
        defensive_leadership=defensive_leadership,
        warning=warning,
    )
