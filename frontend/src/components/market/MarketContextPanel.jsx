import { useEffect, useState } from 'react'
import { api } from '../../api'
import { formatCurrency, formatPercent } from '../../format'
import TrendBadge from './TrendBadge'
import NewsList from './analysis/NewsList'

const REGIME_META = {
  favorable: { label: 'Favorable', tone: 'up' },
  precaucion: { label: 'Precaución', tone: 'warn' },
  evitar: { label: 'Evitar / esperar', tone: 'down' },
}

function MarketContextPanel() {
  const [context, setContext] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setContext(await api.getMarketContext())
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <p className="empty-state">Cargando contexto de mercado…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (!context) return null

  const { indices, vix, regime, news } = context
  const regimeMeta = REGIME_META[regime.verdict] ?? { label: regime.verdict, tone: 'neutral' }

  return (
    <div>
      <div className={`regime-banner regime-banner--${regimeMeta.tone}`}>
        <div className="regime-banner__headline-row">
          <span className={`regime-banner__badge regime-banner__badge--${regimeMeta.tone}`}>{regimeMeta.label}</span>
          <p className="regime-banner__headline">{regime.headline}</p>
        </div>
        <ul className="regime-banner__reasons">
          {regime.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      </div>

      <div className="context-grid">
      <div className="context-card">
        <p className="context-card__title">VIX</p>
        <p className="context-card__big-value">{vix.level !== null ? vix.level.toFixed(1) : '—'}</p>
        <p className="context-card__subtitle">{vix.regime}</p>
        {vix.sma50 !== null && <p className="context-card__hint">Media 50d: {vix.sma50.toFixed(1)}</p>}
        {vix.term_structure && <p className="context-card__hint">Estructura temporal: {vix.term_structure}</p>}
      </div>

      <div className="context-card context-card--wide">
        <p className="context-card__title">Índices de referencia</p>
        <div className="table-scroll">
          <table className="positions-table">
            <thead>
              <tr>
                <th>Índice</th>
                <th className="num">Precio</th>
                <th className="num">1D</th>
                <th className="num">1S</th>
                <th className="num">1M</th>
                <th className="num">3M</th>
                <th className="num">1A</th>
                <th>Tendencia</th>
              </tr>
            </thead>
            <tbody>
              {indices.map((index) => (
                <tr key={index.ticker}>
                  <td className="positions-table__ticker">{index.name}</td>
                  <td className="num">{index.price !== null ? formatCurrency(index.price) : '—'}</td>
                  <td className="num">
                    <span className={index.change_1d >= 0 ? 'delta-up' : 'delta-down'}>
                      {formatPercent(index.change_1d, { signed: true })}
                    </span>
                  </td>
                  <td className="num">
                    <span className={index.change_1w >= 0 ? 'delta-up' : 'delta-down'}>
                      {formatPercent(index.change_1w, { signed: true })}
                    </span>
                  </td>
                  <td className="num">
                    <span className={index.change_1m >= 0 ? 'delta-up' : 'delta-down'}>
                      {formatPercent(index.change_1m, { signed: true })}
                    </span>
                  </td>
                  <td className="num">
                    <span className={index.change_3m >= 0 ? 'delta-up' : 'delta-down'}>
                      {formatPercent(index.change_3m, { signed: true })}
                    </span>
                  </td>
                  <td className="num">
                    <span className={index.change_1y >= 0 ? 'delta-up' : 'delta-down'}>
                      {formatPercent(index.change_1y, { signed: true })}
                    </span>
                  </td>
                  <td>{index.trend && <TrendBadge trend={index.trend} />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="context-card context-card--wide">
        <p className="context-card__title">Noticias generales del mercado (S&amp;P 500)</p>
        <NewsList news={news} emptyMessage="No hay noticias generales disponibles en este momento." />
      </div>
      </div>
    </div>
  )
}

export default MarketContextPanel
