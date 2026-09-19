"""Auditoria del Radar, bloque 9: "rotación contra cartera con topes" - un
paso más allá de `opportunity_cost.py`, que deliberadamente solo señala "hay
una alternativa cuyo gate aprueba en tu sector" sin nunca decidir si la
posición en sí debería venderse (ver su propio docstring: "nunca una segunda
opinión sobre si la posición en sí debería venderse"). Esto SÍ recomienda un
swap concreto (vender X, comprar Y) - pero solo cuando la propia posición ya
es objetivamente débil según `exit_engine.ExitUrgency` (`exit_now`/`reduce`,
YA decidido y persistido por `daily_close.py` en `position_daily_states` -
nunca recalculado aquí) y solo cuando la cartera está llena (rotar tiene
sentido cuando no hay hueco libre, no como sustituto de simplemente añadir
una posición nueva).

Comprar y vender siguen siendo preguntas distintas (CLAUDE.md,
docs/quant_methodology.md §8): esta función NUNCA decide "vender" por su
cuenta - `exit_engine.py` ya lo decidió; esto solo añade la mitad que faltaba,
"¿y con qué la reemplazo?", usando el grado A/B/C del Radar (Parte 5.3, ya
precomputado en `TickerDailyState.grade` - no un score nuevo calculado en
este módulo) como criterio de calidad del reemplazo, con el RS Rating como
desempate, mismo idiom que `opportunity_cost.find_opportunity_cost_notes` ya
usa. Deliberadamente NO el score compuesto del Radar (`setups/scoring.py`):
ese pipeline vive dentro de `GET /market/radar` (percentiles de ATR del
universo del día, penalización de earnings, etc.) y reproducirlo aquí
duplicaría lógica que ya tiene su propio hogar - el grado ya es la señal de
calidad que este mismo módulo (`opportunity_cost.py`) trata como suficiente
para "¿es esta alternativa genuinamente mejor?", y sigue el mismo principio.

Puro cálculo sobre datos ya calculados (`ticker_daily_states`/
`position_daily_states`) - ninguna llamada de red, ningún indicador
recalculado, mismo "sin cómputo en caliente" que el resto de Radar/Hoy."""

from dataclasses import dataclass

from app.core.trading_params import MAX_OPEN_POSITIONS, ROTATION_MAX_SUGGESTIONS_PER_DAY
from app.domain.models.ticker_daily_state import TickerDailyState
from app.services.market_universe import sector_of

# Las únicas dos urgencias que hacen a una posición elegible para salir -
# `exit_engine.ExitUrgency` tiene cinco niveles; tighten_stop/watch/hold no
# son motivo de rotación, son gestión normal de una posición sana.
_WEAK_URGENCIES = ("exit_now", "reduce")
_GRADE_RANK = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True, slots=True)
class RotationSuggestion:
    sell_ticker: str
    sell_reason: str  # cita la urgencia de salida real de exit_engine.py, en español
    buy_ticker: str
    buy_reason: str  # cita el grado y el sector del candidato, en español
    sector: str


def suggest_rotations(
    held_states: list[TickerDailyState],
    urgency_by_ticker: dict[str, str],
    radar_candidates: list[TickerDailyState],
    open_positions_count: int,
    max_open_positions: int = MAX_OPEN_POSITIONS,
    max_suggestions: int = ROTATION_MAX_SUGGESTIONS_PER_DAY,
) -> list[RotationSuggestion]:
    """`held_states`/`radar_candidates`: mismo shape exacto que
    `opportunity_cost.find_opportunity_cost_notes` ya recibe en
    `GET /portfolios/{id}/today` - filas de `ticker_daily_states`, la primera
    para lo que se tiene, la segunda ya filtrada a `gate_passes=True`. El
    sector se deriva con `market_universe.sector_of(ticker)`, igual que
    `opportunity_cost.py` - nunca lee `TickerDailyState.sector` directamente,
    para no depender de que esa columna (Parte 8 de la biblioteca de setups)
    esté poblada en cada fila; el mapeo curado es la misma fuente de verdad
    de todos modos. `urgency_by_ticker` viene de `position_daily_states.urgency`
    (el mismo campo que `PositionDailyStateResponse.urgency` ya expone) - un
    ticker sin entrada en el dict (aún no evaluado por el motor de salida)
    nunca es elegible, nunca se asume débil por defecto.

    Reglas, en orden:
    1. Sin sugerencias si la cartera no está llena (`open_positions_count <
       max_open_positions`) - hay hueco libre, rotar no aplica todavía.
    2. Solo posiciones con urgencia `exit_now`/`reduce` son elegibles para
       salir, `exit_now` antes que `reduce` (más débil primero).
    3. El reemplazo debe estar en el mismo sector, tener gate aprobado y
       grado A/B/C real (nunca `None`) - nunca sugiere vender algo débil por
       un candidato sin evidencia de calidad. Desempate por RS Rating, igual
       que `opportunity_cost.py`.
    4. Nunca reutiliza el mismo candidato para dos sugerencias distintas, ni
       sugiere un ticker que ya se tiene.
    5. Tope duro `max_suggestions` - "aviso, no automatización"."""
    if open_positions_count < max_open_positions:
        return []

    weak = [
        state for state in held_states if urgency_by_ticker.get(state.ticker) in _WEAK_URGENCIES
    ]
    if not weak:
        return []
    weak.sort(key=lambda s: 0 if urgency_by_ticker[s.ticker] == "exit_now" else 1)

    held_tickers = {state.ticker for state in held_states}
    used_candidates: set[str] = set()
    suggestions: list[RotationSuggestion] = []

    for position in weak:
        if len(suggestions) >= max_suggestions:
            break
        sector = sector_of(position.ticker)
        if sector is None:
            continue
        alternatives = [
            c
            for c in radar_candidates
            if c.gate_passes
            and c.ticker not in held_tickers
            and c.ticker not in used_candidates
            and sector_of(c.ticker) == sector
            and c.grade is not None
            and c.grade.get("grade") in _GRADE_RANK
        ]
        if not alternatives:
            continue
        alternatives.sort(
            key=lambda c: (_GRADE_RANK[c.grade["grade"]], c.rs_rating is None, -(c.rs_rating or 0))
        )
        best = alternatives[0]
        used_candidates.add(best.ticker)
        urgency = urgency_by_ticker[position.ticker]
        urgency_label = "salida inmediata" if urgency == "exit_now" else "reducir la posición"
        suggestions.append(
            RotationSuggestion(
                sell_ticker=position.ticker,
                sell_reason=f"el motor de salida marca {urgency_label} en {position.ticker}",
                buy_ticker=best.ticker,
                buy_reason=(
                    f"{best.ticker} tiene grado {best.grade['grade']} y gate aprobado en el mismo "
                    f"sector ({sector})"
                ),
                sector=sector,
            )
        )
    return suggestions
