import {
  ACTION_REQUIRED_URGENCY_TIERS,
  EXIT_URGENCY_ICON,
  EXIT_URGENCY_LABELS,
  exitUrgencyRank,
  formatCurrency,
  formatPercent,
  formatRMultiple,
} from '../format'

// The actual UI payoff of the exit engine (D2/D3 - see exit_engine.py):
// every held position whose exit_urgency is a real action, not just a
// watch, ranked by severity so the single most important thing to do today
// is always the first line - never buried in a table sorted by position
// size. Deliberately built from `riskByTicker` alone (not `positions`), so
// it renders nothing at all while risk is still loading rather than a
// misleading "nothing to do" for a heartbeat.
//
// Cuarta auditoría, recomendación FE-2: hasta ahora este panel solo miraba
// señales por posición - un riesgo agregado real (D12,
// portfolio_construction_service.py) que supera el límite objetivo del 6%
// podía estar activo sin que nada "requerido hoy" lo reflejara, aunque
// `PortfolioConstructionPanel` ya lo mostrara como un panel secundario más
// abajo en el dashboard. `construction` es opcional a propósito - este panel
// sigue funcionando igual (solo con las señales por posición) si todavía no
// ha llegado.
function TodayActionsPanel({ riskByTicker, onNavigateToTicker, construction, currency = 'USD' }) {
  if (!riskByTicker) return null

  const actionable = [...riskByTicker.values()]
    .filter((risk) => ACTION_REQUIRED_URGENCY_TIERS.includes(risk.exit_urgency))
    .sort((a, b) => exitUrgencyRank(a.exit_urgency) - exitUrgencyRank(b.exit_urgency))

  const aggregateRisk = construction?.aggregate_risk
  const aggregateRiskBreached = aggregateRisk?.exceeds_limit === true

  if (actionable.length === 0 && !aggregateRiskBreached) {
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
        {aggregateRiskBreached && (
          <li className="today-actions__item today-actions__item--exit_now">
            <div className="today-actions__headline">
              <span className="today-actions__icon" aria-hidden="true">
                ⚠️
              </span>
              <span className="today-actions__verb">Recortar exposición agregada</span>
              <strong className="today-actions__ticker">Cartera completa</strong>
            </div>
            <ul className="today-actions__reasons">
              <li>
                Riesgo agregado de {formatCurrency(aggregateRisk.total_risk_amount, currency)}
                {aggregateRisk.total_risk_pct_of_capital !== null && (
                  <> ({formatPercent(aggregateRisk.total_risk_pct_of_capital)} del capital)</>
                )}{' '}
                si saltaran todos los stops a la vez - supera el límite objetivo del 6% (ver "Construcción de
                cartera" más abajo para el detalle por posición).
              </li>
            </ul>
          </li>
        )}
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
