import { SIGNAL_LABELS } from '../format'

// See opportunity_cost.py: a held position that isn't broken, but where a
// materially stronger, already-vetted premium candidate exists instead - a
// suggestion to consider, never an instruction to act on automatically.
function SwapSuggestions({ suggestions, onNavigateToTicker }) {
  if (!suggestions || suggestions.length === 0) return null

  return (
    <div className="swap-suggestions">
      <p className="swap-suggestions__title">💡 Coste de oportunidad</p>
      <ul className="swap-suggestions__list">
        {suggestions.map((s) => (
          <li key={`${s.held_ticker}-${s.candidate_ticker}`} className="swap-suggestions__item">
            <strong>{s.held_ticker}</strong> ({SIGNAL_LABELS[s.held_signal] ?? s.held_signal}, puntuación{' '}
            {s.held_score}) podría cambiarse por{' '}
            {onNavigateToTicker ? (
              <button
                type="button"
                className="swap-suggestions__link"
                onClick={() => onNavigateToTicker(s.candidate_ticker)}
              >
                {s.candidate_ticker}
              </button>
            ) : (
              <strong>{s.candidate_ticker}</strong>
            )}{' '}
            ({s.candidate_sector}, puntuación {s.candidate_score.toFixed(1)}) - tiene un setup más fuerte ahora
            mismo, ya validado con el mismo análisis completo de "Analizar activo".
          </li>
        ))}
      </ul>
    </div>
  )
}

export default SwapSuggestions
