import { formatCurrency, formatPercent } from '../format'

// See portfolio_construction_service.py (D12): risk no single position's own
// exit_urgency can ever see - correlated bets, sector concentration, and the
// real money lost if every open stop got hit at once. Segunda auditoría,
// Bloque 2: this service existed, fully tested in isolation, with zero
// callers anywhere in app/api - this panel is that connection.
function PortfolioConstructionPanel({ construction, currency }) {
  if (!construction) return null

  const {
    correlated_pairs: correlatedPairs,
    sector_concentrations: sectorConcentrations,
    concentrated_sectors: concentratedSectors,
    portfolio_volatility_pct: portfolioVolatilityPct,
    volatility_target_pct: volatilityTargetPct,
    suggested_to_trim: suggestedToTrim,
    aggregate_risk: aggregateRisk,
    tickers_without_trade_plan: tickersWithoutTradePlan,
  } = construction

  const concentratedSet = new Set((concentratedSectors ?? []).map((s) => s.sector))
  const hasAnything =
    (sectorConcentrations ?? []).length > 0 || (correlatedPairs ?? []).length > 0 || aggregateRisk

  if (!hasAnything) {
    return (
      <section className="panel">
        <h2>Construcción de cartera</h2>
        <p className="empty-state">Añade posiciones para ver correlación, concentración y riesgo agregado.</p>
      </section>
    )
  }

  return (
    <section className="panel">
      <h2>Construcción de cartera</h2>
      <p className="panel__hint">
        Riesgo a nivel de cartera que ninguna posición individual puede ver por sí sola: apuestas
        correlacionadas, concentración por sector y el dinero real en juego si saltaran todos los stops a la vez.
      </p>

      {aggregateRisk && (
        <div className={`construction-risk-card${aggregateRisk.exceeds_limit ? ' construction-risk-card--over' : ''}`}>
          <p className="construction-risk-card__label">Riesgo agregado si saltan todos los stops</p>
          <p className="construction-risk-card__value">
            {formatCurrency(aggregateRisk.total_risk_amount, currency)}
            {aggregateRisk.total_risk_pct_of_capital !== null && (
              <span> ({formatPercent(aggregateRisk.total_risk_pct_of_capital)} del capital)</span>
            )}
          </p>
          {aggregateRisk.exceeds_limit && (
            <p className="construction-risk-card__warning">
              Supera el límite objetivo (6% del capital) - considera recortar exposición.
            </p>
          )}
          {tickersWithoutTradePlan && tickersWithoutTradePlan.length > 0 && (
            <p className="construction-risk-card__hint">
              Excluidos de este cálculo (sin stop conocido todavía): {tickersWithoutTradePlan.join(', ')}
            </p>
          )}
        </div>
      )}

      {portfolioVolatilityPct !== null && portfolioVolatilityPct !== undefined && (
        <p className="construction-vol-line">
          Volatilidad anualizada de la cartera: <strong>{formatPercent(portfolioVolatilityPct)}</strong>
          {' '}(objetivo: {formatPercent(volatilityTargetPct)})
          {portfolioVolatilityPct > volatilityTargetPct && suggestedToTrim && suggestedToTrim.length > 0 && (
            <>
              {' '}- mayores contribuyentes al riesgo:{' '}
              {suggestedToTrim.map((r) => `${r.ticker} (${formatPercent(r.risk_contribution_pct)})`).join(', ')}
            </>
          )}
        </p>
      )}

      {correlatedPairs && correlatedPairs.length > 0 && (
        <div className="construction-block">
          <h3>Posiciones correlacionadas</h3>
          <ul className="construction-list">
            {correlatedPairs.map((p) => (
              <li key={`${p.ticker_a}-${p.ticker_b}`}>
                <strong>{p.ticker_a}</strong> y <strong>{p.ticker_b}</strong> se mueven correlacionados al{' '}
                {formatPercent(Math.abs(p.correlation))} - una sola apuesta con dos nombres, no dos independientes.
              </li>
            ))}
          </ul>
        </div>
      )}

      {sectorConcentrations && sectorConcentrations.length > 0 && (
        <div className="construction-block">
          <h3>Concentración por sector</h3>
          <ul className="construction-sector-list">
            {sectorConcentrations.map((s) => (
              <li key={s.sector} className={concentratedSet.has(s.sector) ? 'construction-sector--over' : ''}>
                <div className="construction-sector__row">
                  <span>{s.sector}</span>
                  <span>
                    {formatPercent(s.weight_pct)}
                    {concentratedSet.has(s.sector) && ' ⚠️'}
                  </span>
                </div>
                <div className="construction-sector__bar">
                  <div
                    className="construction-sector__bar-fill"
                    style={{ width: `${Math.min(100, s.weight_pct * 100)}%` }}
                  />
                </div>
                <p className="construction-sector__tickers">{s.tickers.join(', ')}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

export default PortfolioConstructionPanel
