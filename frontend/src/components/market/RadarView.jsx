import { useEffect, useState } from 'react'
import { api } from '../../api'
import { formatCurrency, formatRelativeTime } from '../../format'
import { stageLabel } from '../../marketFormat'
import TrendBadge from './TrendBadge'
import RecommendationCard from './analysis/RecommendationCard'

// Reconstruction (2026-09), Fase 5: "qué está a punto de disparar una
// entrada" (Parte 0, pregunta 2) - a pure read over GET /market/radar
// (daily_close.py's own precomputed ticker_daily_states), never a live
// universe scan. Replaces the general "A revisar" watchlist tab for this
// purpose (see Watchlist.jsx's own note); that tab stays alongside this one
// for now - retiring watchlist_service.py outright is separate, deferred
// work (docs/quant_methodology.md §25).

// RadarItemResponse flattens what RecommendationCard expects as one nested
// `gate` object (GateResultResponse) - same shape as the "Analizar activo"
// card, just adapted at the call site rather than duplicating that card's
// rendering logic here.
function toGate(item) {
  return {
    passes: item.gate_passes,
    conditions: item.gate_conditions,
    entry_trigger: item.entry_trigger,
    stop_and_target: item.stop_and_target,
  }
}

function RadarRow({ item, onNavigateToTicker }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className={`radar-row ${item.gate_passes ? 'radar-row--pass' : 'radar-row--pending'}`}>
      <div className="radar-row__summary">
        <div className="radar-row__identity">
          {onNavigateToTicker ? (
            <button
              type="button"
              className="positions-table__ticker positions-table__ticker--link"
              onClick={() => onNavigateToTicker(item.ticker)}
              title="Ver análisis completo"
            >
              {item.ticker}
            </button>
          ) : (
            <span className="positions-table__ticker">{item.ticker}</span>
          )}
          <TrendBadge trend={item.trend} />
          <span className="watchlist-card__industry">{stageLabel(item.stage)}</span>
        </div>

        <div className="radar-row__stats">
          <span>{formatCurrency(item.price, item.currency)}</span>
          {item.rs_rating !== null && <span>RS {item.rs_rating}</span>}
          <span className={`badge ${item.gate_passes ? 'badge--buy' : 'badge--neutral'}`}>
            {item.gate_passes ? 'Gate aprobado' : 'Gate no aprobado'}
          </span>
          {item.entry_trigger && (
            <span className={item.entry_trigger.already_triggered ? 'delta-up' : 'watchlist-card__industry'}>
              {item.entry_trigger.already_triggered ? 'Disparador ya activado' : 'Vigilando disparador'}
            </span>
          )}
          <button type="button" className="timeframe-tab" onClick={() => setExpanded((v) => !v)}>
            {expanded ? 'Ocultar detalle' : 'Ver detalle'}
          </button>
        </div>
      </div>

      {expanded && <RecommendationCard gate={toGate(item)} currency={item.currency} />}
    </div>
  )
}

function RadarView({ onNavigateToTicker, region }) {
  const [items, setItems] = useState([])
  const [computedAt, setComputedAt] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const body = await api.getRadar({ region })
        setItems(body.items)
        setComputedAt(body.computed_at)
        setError(null)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  const relative = formatRelativeTime(computedAt)
  const passing = [...items].filter((i) => i.gate_passes).sort((a, b) => a.ticker.localeCompare(b.ticker))
  const pending = [...items].filter((i) => !i.gate_passes).sort((a, b) => a.ticker.localeCompare(b.ticker))

  return (
    <div>
      {relative && <p className="refresh-bar__label">Calculado por el cierre diario · actualizado {relative}</p>}

      {loading ? (
        <p className="empty-state">Cargando radar…</p>
      ) : error ? (
        <div className="banner banner--error">{error}</div>
      ) : items.length === 0 ? (
        <p className="empty-state">
          Sin candidatos todavía - el radar se rellena con el cierre diario (`daily_close.py`); si esta región nunca
          ha corrido, vuelve más tarde.
        </p>
      ) : (
        <div className="radar-list">
          {passing.length > 0 && (
            <section className="watchlist-section">
              <h4 className="watchlist-section__title">Gate aprobado - entrada válida hoy</h4>
              {passing.map((item) => (
                <RadarRow key={item.ticker} item={item} onNavigateToTicker={onNavigateToTicker} />
              ))}
            </section>
          )}
          {pending.length > 0 && (
            <section className="watchlist-section">
              <h4 className="watchlist-section__title">Acercándose - disparador vigilado, gate todavía no aprobado</h4>
              {pending.map((item) => (
                <RadarRow key={item.ticker} item={item} onNavigateToTicker={onNavigateToTicker} />
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  )
}

export default RadarView
