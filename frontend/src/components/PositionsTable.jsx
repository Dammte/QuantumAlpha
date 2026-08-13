import { SIGNAL_LABELS, formatCurrency, formatPercent, garchRegimeLabel } from '../format'

function backtestSummary(backtest) {
  if (!backtest) return null
  const test = backtest.significance_tests.find((t) => t.comparison === 'comprar vs evitar')
  if (!test) return 'historial insuficiente para validar el sistema en este activo'
  if (test.significant_at_5pct && test.mean_difference > 0) return 'ventaja histórica confirmada en este activo'
  if (test.significant_at_5pct && test.mean_difference <= 0) return 'sin ventaja histórica en este activo - tratar con cautela'
  return 'sin diferencia estadísticamente significativa en el historial'
}

function riskTooltip(risk) {
  const parts = [`Puntuación: ${risk.score}`, risk.reasons.join(' · ')]
  if (risk.signals?.entry_timing) parts.push(risk.signals.entry_timing.label)
  if (risk.signals?.garch) parts.push(garchRegimeLabel(risk.signals.garch.regime))
  const backtest = backtestSummary(risk.signals?.backtest)
  if (backtest) parts.push(backtest)
  return parts.join(' · ')
}

function PositionsTable({ positions, colorScale, totalMarketValue, riskByTicker, riskLoading, onNavigateToTicker }) {
  if (positions.length === 0) {
    return <p className="empty-state">Esta cartera todavía no tiene posiciones. Añade una transacción para empezar.</p>
  }

  const sorted = [...positions].sort((a, b) => (b.market_value_base ?? 0) - (a.market_value_base ?? 0))
  // Shown as soon as we're trying to fetch a recommendation, not only once it
  // arrives - the quant suite behind it can take a while on a cold cache, and
  // hiding the whole column until then reads as "no recommendation exists"
  // instead of "still calculating it".
  const showRiskColumn = riskLoading || Boolean(riskByTicker)

  return (
    <div className="table-scroll">
      <table className="positions-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th className="num">Cantidad</th>
            <th className="num">Precio medio</th>
            <th className="num">Precio actual</th>
            <th className="num">Variación hoy</th>
            <th className="num">Valor de mercado</th>
            <th className="num" title="Peso sobre el capital invertido (no incluye la liquidez disponible)">
              Peso invertido
            </th>
            <th className="num">Ganancia total</th>
            {showRiskColumn && <th>Señal de tendencia</th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map((position) => {
            const priceUnavailable = position.current_price === null
            const isUp = (position.unrealized_pnl ?? 0) >= 0
            const dayIsUp = (position.day_change ?? 0) >= 0
            const swatch = colorScale.get(position.ticker)
            const weight =
              totalMarketValue && position.market_value_base !== null ? position.market_value_base / totalMarketValue : null
            const risk = riskByTicker?.get(position.ticker)
            const totalGain = (position.unrealized_pnl ?? 0) + position.realized_pnl
            return (
              <tr key={position.ticker}>
                <td>
                  <span className="ticker-swatch" style={{ '--swatch': swatch.css }} />
                  {onNavigateToTicker ? (
                    <button
                      type="button"
                      className="positions-table__ticker positions-table__ticker--link"
                      onClick={() => onNavigateToTicker(position.ticker)}
                      title="Ver análisis completo"
                    >
                      {position.ticker}
                    </button>
                  ) : (
                    <span className="positions-table__ticker">{position.ticker}</span>
                  )}
                </td>
                <td className="num">{position.quantity}</td>
                <td className="num">{formatCurrency(position.average_cost, position.currency)}</td>
                <td className="num">
                  {priceUnavailable ? (
                    <span className="positions-table__unavailable" title="No se pudo obtener una cotización en vivo para este ticker ahora mismo">
                      No disponible
                    </span>
                  ) : (
                    formatCurrency(position.current_price, position.currency)
                  )}
                </td>
                <td className="num">
                  {position.day_change === null ? (
                    '—'
                  ) : (
                    <span className={dayIsUp ? 'delta-up' : 'delta-down'}>
                      {formatCurrency(position.day_change, position.currency)}
                      <span className="positions-table__pct"> ({formatPercent(position.day_change_pct, { signed: true })})</span>
                    </span>
                  )}
                </td>
                <td className="num">
                  {position.market_value === null ? '—' : formatCurrency(position.market_value, position.currency)}
                </td>
                <td className="num">{weight === null ? '—' : formatPercent(weight)}</td>
                <td className={`num ${isUp ? 'delta-up' : 'delta-down'}`}>
                  {formatCurrency(totalGain, position.currency)}
                  {position.realized_pnl !== 0 && (
                    <span
                      className="positions-table__pct"
                      title={`No realizado: ${formatCurrency(position.unrealized_pnl ?? 0, position.currency)} · Realizado (ventas parciales): ${formatCurrency(position.realized_pnl, position.currency)}`}
                    >
                      {' '}
                      (incl. {formatCurrency(position.realized_pnl, position.currency)} realizado)
                    </span>
                  )}
                </td>
                {showRiskColumn && (
                  <td>
                    {risk ? (
                      <span className={`signal-badge signal-badge--${risk.signal}`} title={riskTooltip(risk)}>
                        {SIGNAL_LABELS[risk.signal] ?? risk.signal}
                        <span className="signal-badge__score"> ({risk.score >= 0 ? '+' : ''}{risk.score})</span>
                        {risk.signal === 'add_candidate' && risk.signals?.entry_timing?.status === 'extended' && (
                          <span title="Setup válido, pero muy extendido - más prudente esperar un retroceso antes de añadir">
                            {' '}
                            ⚠️
                          </span>
                        )}
                      </span>
                    ) : riskLoading ? (
                      <span className="signal-badge signal-badge--pending" title="Calculando la señal técnica completa (tendencia, GARCH, Markov, backtest)…">
                        Calculando…
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default PositionsTable
