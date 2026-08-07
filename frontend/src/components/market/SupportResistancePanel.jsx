import { useEffect, useState } from 'react'
import { api } from '../../api'
import { formatCurrency, formatPercent } from '../../format'

function LevelBadgeRow({ level, currency }) {
  return (
    <span className={`badge ${level.kind === 'resistance' ? 'badge--sell' : 'badge--buy'}`}>
      {level.kind === 'resistance' ? 'Resistencia' : 'Soporte'} {formatCurrency(level.price, currency)} (
      {formatPercent(level.distance_pct, { signed: true })})
    </span>
  )
}

function PortfolioWatch() {
  const [portfolios, setPortfolios] = useState([])
  const [portfolioId, setPortfolioId] = useState(null)
  const [risk, setRisk] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const list = await api.listPortfolios()
        setPortfolios(list)
        if (list.length > 0) setPortfolioId(list[0].id)
        else setLoading(false)
      } catch (err) {
        setError(err.message)
        setLoading(false)
      }
    }
    load()
  }, [])

  useEffect(() => {
    if (!portfolioId) return
    async function load() {
      setLoading(true)
      try {
        setRisk(await api.getPortfolioRisk(portfolioId))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [portfolioId])

  if (portfolios.length === 0 && !loading) {
    return <p className="empty-state">Crea una cartera en "Mi Cartera" para ver aquí sus soportes y resistencias.</p>
  }

  return (
    <div>
      {portfolios.length > 1 && (
        <label className="filter" style={{ marginBottom: 16 }}>
          <span>Cartera</span>
          <select value={portfolioId ?? ''} onChange={(e) => setPortfolioId(Number(e.target.value))}>
            {portfolios.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      )}

      {error && <div className="banner banner--error">{error}</div>}
      {loading ? (
        <p className="empty-state">Cargando posiciones…</p>
      ) : !risk || risk.positions.length === 0 ? (
        <p className="empty-state">Esta cartera todavía no tiene posiciones.</p>
      ) : (
        <div className="table-scroll">
          <table className="positions-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th className="num">Precio</th>
                <th>Soporte más cercano</th>
                <th>Resistencia más cercana</th>
                <th>Señal</th>
              </tr>
            </thead>
            <tbody>
              {risk.positions.map((p) => (
                <tr key={p.ticker}>
                  <td className="positions-table__ticker">{p.ticker}</td>
                  <td className="num">{formatCurrency(p.price, p.currency)}</td>
                  <td>{p.nearest_support ? <LevelBadgeRow level={p.nearest_support} currency={p.currency} /> : '—'}</td>
                  <td>
                    {p.nearest_resistance ? <LevelBadgeRow level={p.nearest_resistance} currency={p.currency} /> : '—'}
                  </td>
                  <td>
                    <span className={`signal-badge signal-badge--${p.signal}`}>
                      {
                        {
                          exit_warning: 'Salir / vigilar',
                          add_candidate: 'Aumentar',
                          watch: 'Vigilar',
                          hold: 'Mantener',
                        }[p.signal]
                      }
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function ProximityScreener({ region }) {
  const [threshold, setThreshold] = useState(0.03)
  const [matches, setMatches] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setMatches(await api.getLevelsProximity({ region, threshold }))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region, threshold])

  return (
    <div>
      <label className="filter filter--inline" style={{ marginBottom: 16 }}>
        <span>Umbral de proximidad</span>
        <select value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}>
          <option value={0.01}>1%</option>
          <option value={0.02}>2%</option>
          <option value={0.03}>3%</option>
          <option value={0.05}>5%</option>
        </select>
      </label>

      {error && <div className="banner banner--error">{error}</div>}
      {loading ? (
        <p className="empty-state">Buscando niveles cercanos…</p>
      ) : matches.length === 0 ? (
        <p className="empty-state">Ningún activo del universo está cerca de un nivel ahora mismo.</p>
      ) : (
        <div className="table-scroll">
          <table className="positions-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Sector</th>
                <th className="num">Precio</th>
                <th>Nivel</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((m) => (
                <tr key={m.ticker}>
                  <td className="positions-table__ticker">{m.ticker}</td>
                  <td>{m.sector}</td>
                  <td className="num">{formatCurrency(m.price, m.currency)}</td>
                  <td>
                    <LevelBadgeRow level={m.level} currency={m.currency} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function TickerSearch() {
  const [ticker, setTicker] = useState('AAPL')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const search = async (value) => {
    const symbol = value.trim().toUpperCase()
    if (!symbol) return
    setLoading(true)
    setError(null)
    try {
      setResult(await api.getSupportResistance(symbol))
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    search(ticker)
  }

  const sortedLevels = result ? [...result.levels].sort((a, b) => b.price - a.price) : []

  return (
    <div>
      <form className="filters-row" onSubmit={handleSubmit}>
        <label className="filter">
          <span>Ticker</span>
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} placeholder="AAPL" />
        </label>
        <button type="submit" className="button-primary" disabled={loading} style={{ alignSelf: 'flex-end' }}>
          {loading ? 'Buscando…' : 'Analizar'}
        </button>
      </form>

      {error && <div className="banner banner--error">{error}</div>}

      {result && (
        <div className="levels-panel">
          <p className="levels-panel__price">
            {result.ticker} · Precio actual: <strong>{formatCurrency(result.price, result.currency)}</strong>
          </p>
          {sortedLevels.length === 0 ? (
            <p className="empty-state">No se detectaron niveles claros de soporte/resistencia en el rango analizado.</p>
          ) : (
            <ul className="levels-list">
              {sortedLevels.map((level) => (
                <li key={`${level.kind}-${level.price}`} className={`levels-list__item levels-list__item--${level.kind}`}>
                  <span className={`badge ${level.kind === 'resistance' ? 'badge--sell' : 'badge--buy'}`}>
                    {level.kind === 'resistance' ? 'Resistencia' : 'Soporte'}
                  </span>
                  <span className="levels-list__price">{formatCurrency(level.price, result.currency)}</span>
                  <span className="levels-list__distance">{formatPercent(level.distance_pct, { signed: true })}</span>
                  <span className="levels-list__strength" title="Número de pivotes agrupados en este nivel">
                    {'●'.repeat(Math.min(level.strength, 5))}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function SupportResistancePanel({ region }) {
  return (
    <div className="levels-dashboard">
      <div className="panel panel--nested">
        <h3>Tu cartera: soportes y resistencias próximas</h3>
        <PortfolioWatch />
      </div>
      <div className="panel panel--nested">
        <h3>Activos del universo cerca de un nivel</h3>
        <ProximityScreener region={region} />
      </div>
      <div className="panel panel--nested">
        <h3>Buscar un ticker concreto</h3>
        <TickerSearch />
      </div>
    </div>
  )
}

export default SupportResistancePanel
