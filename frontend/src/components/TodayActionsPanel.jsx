import {
  ACTION_REQUIRED_URGENCY_TIERS,
  EXIT_URGENCY_ICON,
  EXIT_URGENCY_LABELS,
  exitUrgencyRank,
  formatRMultiple,
} from '../format'

// The actual UI payoff of the exit engine (D2/D3 - see exit_engine.py):
// every held position whose exit_urgency is a real action, not just a
// watch, ranked by severity so the single most important thing to do today
// is always the first line - never buried in a table sorted by position
// size. Deliberately built from `riskByTicker` alone (not `positions`), so
// it renders nothing at all while risk is still loading rather than a
// misleading "nothing to do" for a heartbeat.
function TodayActionsPanel({ riskByTicker, onNavigateToTicker }) {
  if (!riskByTicker) return null

  const actionable = [...riskByTicker.values()]
    .filter((risk) => ACTION_REQUIRED_URGENCY_TIERS.includes(risk.exit_urgency))
    .sort((a, b) => exitUrgencyRank(a.exit_urgency) - exitUrgencyRank(b.exit_urgency))

  if (actionable.length === 0) {
    return (
      <section className="panel today-actions today-actions--empty">
        <h2>Acciones requeridas hoy</h2>
        <p className="empty-state">
          Ninguna posición necesita una acción hoy - las tesis siguen intactas o, como mucho, en vigilancia.
        </p>
      </section>
    )
  }

  return (
    <section className="panel today-actions">
      <h2>Acciones requeridas hoy</h2>
      <ul className="today-actions__list">
        {actionable.map((risk) => (
          <li key={risk.ticker} className={`today-actions__item today-actions__item--${risk.exit_urgency}`}>
            <div className="today-actions__headline">
              <span className="today-actions__icon" aria-hidden="true">
                {EXIT_URGENCY_ICON[risk.exit_urgency]}
              </span>
              <span className="today-actions__verb">{EXIT_URGENCY_LABELS[risk.exit_urgency]}</span>
              {onNavigateToTicker ? (
                <button
                  type="button"
                  className="today-actions__ticker"
                  onClick={() => onNavigateToTicker(risk.ticker)}
                  title="Ver análisis completo"
                >
                  {risk.ticker}
                </button>
              ) : (
                <strong className="today-actions__ticker">{risk.ticker}</strong>
              )}
              {risk.r_multiple !== null && (
                <span className="today-actions__r" title="Múltiplo de R actual (ganancia/pérdida sobre el riesgo inicial)">
                  {formatRMultiple(risk.r_multiple)}
                </span>
              )}
            </div>
            {risk.exit_reasons.length > 0 && (
              <ul className="today-actions__reasons">
                {risk.exit_reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

export default TodayActionsPanel
