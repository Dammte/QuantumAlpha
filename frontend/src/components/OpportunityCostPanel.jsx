// Reconstruction (2026-09), Fase 5 (resto): GET /portfolios/{id}/today's
// opportunity_cost field - see opportunity_cost.py's own docstring on the
// backend. Never a second opinion on whether to sell `held_ticker` (that's
// exit_engine.py's job alone, surfaced by TodayActionsPanel above) - purely
// "a same-sector Radar candidate is passing its gate today while this
// holding isn't". Renders nothing when there's nothing to flag.
function OpportunityCostPanel({ notes, onNavigateToTicker }) {
  if (!notes || notes.length === 0) return null

  return (
    <section className="panel opportunity-cost">
      <h2>Coste de oportunidad</h2>
      <p className="opportunity-cost__hint">
        Estos activos de tu cartera no aprueban su propio gate hoy, mientras otro de su mismo sector en el
        Radar sí lo aprueba - no es una señal de venta, solo contexto de qué más está pasando en ese sector.
      </p>
      <ul className="opportunity-cost__list">
        {notes.map((note) => (
          <li key={`${note.held_ticker}-${note.alternative_ticker}`} className="opportunity-cost__item">
            <button
              type="button"
              className="positions-table__ticker positions-table__ticker--link"
              onClick={() => onNavigateToTicker(note.held_ticker)}
            >
              {note.held_ticker}
            </button>
            <span className="opportunity-cost__arrow">→</span>
            <button
              type="button"
              className="positions-table__ticker positions-table__ticker--link"
              onClick={() => onNavigateToTicker(note.alternative_ticker)}
            >
              {note.alternative_ticker}
            </button>
            <span className="opportunity-cost__detail">
              {note.sector}
              {note.alternative_rs_rating !== null && <> · RS {note.alternative_rs_rating}</>}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}

export default OpportunityCostPanel
