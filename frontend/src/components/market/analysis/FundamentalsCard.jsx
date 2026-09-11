import { formatPercent } from '../../../format'

function formatMarketCap(value) {
  if (value === null || value === undefined) return '—'
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} T`
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`
  return value.toLocaleString('en-US')
}

function FundamentalsCard({ fundamentals }) {
  if (!fundamentals) return <p className="empty-state">Datos fundamentales no disponibles para este ticker.</p>

  return (
    <div className="fundamentals-grid">
      <div>
        <p className="fundamentals-grid__label">Capitalización</p>
        <p className="fundamentals-grid__value">{formatMarketCap(fundamentals.market_cap)}</p>
      </div>
      <div>
        <p className="fundamentals-grid__label">PER (trailing / forward)</p>
        <p className="fundamentals-grid__value">
          {fundamentals.trailing_pe?.toFixed(1) ?? '—'} / {fundamentals.forward_pe?.toFixed(1) ?? '—'}
        </p>
      </div>
      <div>
        <p className="fundamentals-grid__label">Dividendo (yield)</p>
        <p className="fundamentals-grid__value">
          {fundamentals.dividend_yield !== null ? formatPercent(fundamentals.dividend_yield / 100) : '—'}
        </p>
      </div>
      <div>
        <p className="fundamentals-grid__label">Beta</p>
        <p className="fundamentals-grid__value">{fundamentals.beta?.toFixed(2) ?? '—'}</p>
      </div>
      <div>
        <p className="fundamentals-grid__label">Volumen medio</p>
        <p className="fundamentals-grid__value">
          {fundamentals.average_volume ? formatMarketCap(fundamentals.average_volume) : '—'}
        </p>
      </div>
      <div>
        <p className="fundamentals-grid__label">Crecimiento de ingresos (interanual)</p>
        <p className="fundamentals-grid__value">
          {fundamentals.revenue_growth !== null ? (
            <span className={fundamentals.revenue_growth >= 0 ? 'delta-up' : 'delta-down'}>
              {formatPercent(fundamentals.revenue_growth, { signed: true })}
            </span>
          ) : (
            '—'
          )}
        </p>
      </div>
      <div>
        <p className="fundamentals-grid__label">Margen neto</p>
        <p className="fundamentals-grid__value">
          {fundamentals.profit_margins !== null ? (
            <span className={fundamentals.profit_margins >= 0 ? 'delta-up' : 'delta-down'}>
              {formatPercent(fundamentals.profit_margins, { signed: true })}
            </span>
          ) : (
            '—'
          )}
        </p>
      </div>
      <div>
        <p className="fundamentals-grid__label">Deuda / Patrimonio</p>
        <p className="fundamentals-grid__value">
          {fundamentals.debt_to_equity !== null ? `${fundamentals.debt_to_equity.toFixed(0)}%` : '—'}
        </p>
      </div>
    </div>
  )
}

export default FundamentalsCard
