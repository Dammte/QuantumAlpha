import { useEffect, useState } from 'react'
import { api } from '../../api'
import { formatPercent } from '../../format'

function SectorForecastRow({ forecast, onNavigateToTicker }) {
  const tone = !forecast.has_statistical_structure ? 'neutral' : forecast.forecast_21d_return >= 0 ? 'up' : 'down'

  return (
    <div className="watchlist-card">
      <div className="watchlist-card__header">
        <span className="positions-table__ticker">{forecast.sector}</span>
        <span className={`sector-tier-badge sector-tier-badge--${tone}`}>
          {forecast.has_statistical_structure
            ? `${formatPercent(forecast.forecast_21d_return, { signed: true })} a 21 días`
            : 'Sin señal estadística clara'}
        </span>
      </div>
      <div className="watchlist-card__stats">
        <span>Estado actual: {forecast.current_state_label}</span>
        <span>Proyección a 5 días: {formatPercent(forecast.forecast_5d_return, { signed: true })}</span>
        <span>Prob. de cierre alcista a 21 días: {formatPercent(forecast.prob_bullish_21d)}</span>
      </div>
      {forecast.top_stocks.length > 0 && (
        <p className="watchlist-card__sector-tag">
          Líderes actuales del sector:{' '}
          {forecast.top_stocks.map((ticker, i) => (
            <span key={ticker}>
              {i > 0 && ', '}
              {onNavigateToTicker ? (
                <button
                  type="button"
                  className="positions-table__ticker positions-table__ticker--link"
                  onClick={() => onNavigateToTicker(ticker)}
                >
                  {ticker}
                </button>
              ) : (
                ticker
              )}
            </span>
          ))}
        </p>
      )}
    </div>
  )
}

// Forward-looking counterpart to SectorStrength (which only shows how each
// sector has *already* performed): fits a Markov chain per sector ETF's own
// history to project where it's statistically likely to head over the next
// 5-21 trading days - the same model "Analizar activo" already applies
// per-stock, aimed at a sector index instead. See sectors/forecast in
// market_screener_service.py.
function SectorForecastCard({ region, onNavigateToTicker }) {
  const [forecast, setForecast] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setForecast(await api.getSectorForecast({ region }))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  if (loading) return <p className="empty-state">Proyectando sectores…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (forecast.length === 0) return null

  return (
    <section className="panel panel--nested">
      <h3>Sectores a vigilar - proyección a corto/medio plazo</h3>
      <p className="empty-state" style={{ marginBottom: 16 }}>
        Que un sector se haya comportado bien no garantiza que lo siga haciendo - esto es distinto de la fuerza
        relativa de arriba: proyecta hacia dónde tiende estadísticamente cada sector en los próximos 5 y 21 días
        de mercado, ajustando una cadena de Markov al histórico de su propio ETF. Cuando ese historial no muestra
        un patrón distinguible del puro azar, se marca como "sin señal estadística clara" en vez de forzar una
        proyección sin base real - los sectores con señal genuina aparecen primero.
      </p>
      <div className="watchlist-grid">
        {forecast.map((f) => (
          <SectorForecastRow key={f.sector} forecast={f} onNavigateToTicker={onNavigateToTicker} />
        ))}
      </div>
    </section>
  )
}

export default SectorForecastCard
