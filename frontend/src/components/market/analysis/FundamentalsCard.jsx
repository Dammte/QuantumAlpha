import { formatCurrency, formatPercent } from '../../../format'

const RECOMMENDATION_LABELS = {
  strong_buy: 'Compra fuerte',
  buy: 'Compra',
  hold: 'Mantener',
  underperform: 'Bajo rendimiento',
  sell: 'Venta',
}

function formatMarketCap(value) {
  if (value === null || value === undefined) return '—'
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} T`
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`
  return value.toLocaleString('en-US')
}

function FundamentalsCard({ fundamentals, price, currency }) {
  if (!fundamentals) return <p className="empty-state">Datos fundamentales no disponibles para este ticker.</p>

  const targetUpside = fundamentals.analyst_target_mean_price ? fundamentals.analyst_target_mean_price / price - 1 : null

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
        <p className="fundamentals-grid__label">Consenso analistas</p>
        <p className="fundamentals-grid__value">
          {RECOMMENDATION_LABELS[fundamentals.analyst_recommendation] ?? fundamentals.analyst_recommendation ?? '—'}
          {fundamentals.analyst_opinion_count ? ` (${fundamentals.analyst_opinion_count})` : ''}
        </p>
      </div>
      <div>
        <p className="fundamentals-grid__label">Precio objetivo (analistas)</p>
        <p className="fundamentals-grid__value">
          {formatCurrency(fundamentals.analyst_target_mean_price, currency)}
          {targetUpside !== null && (
            <span className={targetUpside >= 0 ? 'delta-up' : 'delta-down'}> ({formatPercent(targetUpside, { signed: true })})</span>
          )}
        </p>
      </div>
    </div>
  )
}

export default FundamentalsCard
