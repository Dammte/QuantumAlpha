import { useEffect, useState } from 'react'
import { api } from '../../api'
import { formatCurrency, formatPercent } from '../../format'
import TrendBadge from './TrendBadge'
import NewsList from './analysis/NewsList'

const COMPONENT_LABELS = {
  momentum: 'Momentum (SPX vs MA125)',
  strength: 'Fuerza (máx/mín 52s en el universo)',
  breadth: 'Amplitud (% sobre MA50)',
  volatility: 'Volatilidad (VIX vs su media)',
  safe_haven: 'Refugio seguro (acciones vs bonos)',
  junk_bond_demand: 'Apetito por riesgo (high yield vs grado inversión)',
}

const REGIME_META = {
  favorable: { label: 'Favorable', tone: 'up' },
  precaucion: { label: 'Precaución', tone: 'warn' },
  evitar: { label: 'Evitar / esperar', tone: 'down' },
}

function fearGreedTone(score) {
  if (score >= 56) return 'up'
  if (score <= 44) return 'down'
  return 'neutral'
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

  const { indices, vix, fear_greed: fearGreed, liquidity, regime, news, macro } = context
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
        <p className="context-card__title">Fear &amp; Greed</p>
        <p className={`context-card__big-value context-card__big-value--${fearGreedTone(fearGreed.score)}`}>
          {fearGreed.score.toFixed(0)}
        </p>
        <p className="context-card__subtitle">{fearGreed.label}</p>
        <ul className="context-card__components">
          {Object.entries(fearGreed.components).map(([key, value]) => (
            <li key={key}>
              <span>{COMPONENT_LABELS[key] ?? key}</span>
              <span>{value.toFixed(0)}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="context-card">
        <p className="context-card__title">VIX</p>
        <p className="context-card__big-value">{vix.level !== null ? vix.level.toFixed(1) : '—'}</p>
        <p className="context-card__subtitle">{vix.regime}</p>
        {vix.sma50 !== null && <p className="context-card__hint">Media 50d: {vix.sma50.toFixed(1)}</p>}
        {vix.term_structure && <p className="context-card__hint">Estructura temporal: {vix.term_structure}</p>}
      </div>

      <div className="context-card">
        <p className="context-card__title">Liquidez (proxy dólar)</p>
        <p className={`context-card__big-value ${liquidity.headwind ? 'delta-down' : 'delta-up'}`}>
          {liquidity.headwind ? 'Viento en contra' : 'Viento a favor'}
        </p>
        <p className="context-card__subtitle">{liquidity.trend}</p>
        <p className="context-card__hint">Proxy: {liquidity.proxy_ticker}</p>
      </div>

      {macro && (
        <div className="context-card">
          <p className="context-card__title">Macro (FRED)</p>
          <p className={`context-card__big-value ${macro.yield_curve_inverted ? 'delta-down' : 'delta-up'}`}>
            {macro.yield_curve_spread !== null ? `${macro.yield_curve_spread.toFixed(2)} pp` : '—'}
          </p>
          <p className="context-card__subtitle">
            Curva 10a-2a{macro.yield_curve_inverted ? ' · invertida' : ''}
          </p>
          {macro.yield_curve_date && (
            <p className="context-card__hint">Dato: {macro.yield_curve_date}</p>
          )}
          <ul className="context-card__components">
            <li>
              <span>Desempleo</span>
              <span>{macro.unemployment_rate !== null ? formatPercent(macro.unemployment_rate / 100) : '—'}</span>
            </li>
            <li>
              <span>IPC interanual</span>
              <span>{macro.cpi_yoy_change !== null ? formatPercent(macro.cpi_yoy_change / 100, { signed: true }) : '—'}</span>
            </li>
          </ul>
        </div>
      )}

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
