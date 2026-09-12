"""Reconstruction (2026-09), Fase 5 (resto): "¿podría estar perdiendo una
oportunidad mejor?" - compara cada posición cuyo gate de entrada no aprueba
hoy contra el Radar del mismo sector, usando solo hechos ya calculados
(`gate_passes`, `rs_rating`) - nunca un puntaje nuevo, y nunca una segunda
opinión sobre si la posición en sí debería venderse (eso sigue siendo
trabajo exclusivo de `exit_engine.py` - comprar y vender son preguntas
distintas, ver docs/quant_methodology.md §8).

Reemplaza, en espíritu, al `opportunity_cost.py` original (retirado en la
Fase 1 junto con la watchlist "Premium" de 3 niveles que alimentaba - ver
docs/quant_methodology.md §25) - deliberadamente pequeño: una comparación
booleana (¿el gate de un ticker de tu mismo sector aprueba hoy, mientras el
tuyo no?), no una puntuación de qué tan buena es la alternativa. La fuente
de candidatos es el propio Radar (`ticker_daily_states` con `gate_passes`),
nunca una lista separada."""

from dataclasses import dataclass

from app.domain.models.ticker_daily_state import TickerDailyState
from app.services.market_universe import sector_of


@dataclass(frozen=True, slots=True)
class OpportunityCostNote:
    held_ticker: str
    sector: str
    alternative_ticker: str
    alternative_rs_rating: int | None


def find_opportunity_cost_notes(
    held_states: list[TickerDailyState], radar_candidates: list[TickerDailyState]
) -> list[OpportunityCostNote]:
    """One note per (holding, alternative) pair: a holding whose own gate
    does not pass today, and an already-passing Radar candidate in the same
    curated sector. `radar_candidates` is expected to already be filtered to
    `gate_passes=True` rows (the same ones `GET /market/radar`'s "Gate
    aprobado" section shows) - this function never re-derives that filter
    itself, so it stays a pure comparison, not a second read of the gate.

    Sorted by the alternative's RS Rating (missing last) within each
    holding - not an invented score, just the one already-computed field
    this project already treats as an ordering criterion for "which leader
    is stronger" (same idiom `market_screener_service.py`'s own sort uses).
    A holding whose sector isn't in the curated `market_universe.py` mapping
    is skipped rather than guessed at."""
    notes: list[OpportunityCostNote] = []
    for held in held_states:
        if held.gate_passes:
            continue
        held_sector = sector_of(held.ticker)
        if held_sector is None:
            continue
        alternatives = [
            candidate
            for candidate in radar_candidates
            if candidate.gate_passes
            and candidate.ticker != held.ticker
            and sector_of(candidate.ticker) == held_sector
        ]
        alternatives.sort(key=lambda c: (c.rs_rating is None, -(c.rs_rating or 0)))
        notes.extend(
            OpportunityCostNote(
                held_ticker=held.ticker,
                sector=held_sector,
                alternative_ticker=alt.ticker,
                alternative_rs_rating=alt.rs_rating,
            )
            for alt in alternatives
        )
    return notes
