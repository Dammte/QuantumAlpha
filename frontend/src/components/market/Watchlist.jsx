import { useEffect, useState } from 'react'
import { api } from '../../api'
import { formatCurrency, formatPercent } from '../../format'
import TrendBadge from './TrendBadge'
import PremiumWatchlist from './PremiumWatchlist'
import RefreshBar from '../RefreshBar'

const HORIZONS = [
  { key: '', label: 'Todos' },
  { key: 'short', label: 'Corto plazo' },
  { key: 'medium', label: 'Medio plazo' },
  { key: 'long', label: 'Largo plazo' },
]

const HORIZON_LABELS = { short: 'Corto', medium: 'Medio', long: 'Largo' }

// Mirrors the top/bottom-3-of-11 thresholds sector_rotation_service.py uses to
// call a sector a "leader" or "laggard".
const STRONG_SECTOR_THRESHOLD = 70
const WEAK_SECTOR_THRESHOLD = 30

function sectorTier(rank) {
  if (rank === null || rank === undefined) return null
  if (rank >= STRONG_SECTOR_THRESHOLD) return 'strong'
  if (rank <= WEAK_SECTOR_THRESHOLD) return 'weak'
  return 'neutral'
}

const SECTOR_TIER_META = {
  strong: { label: 'Sector fuerte', tone: 'up' },
  weak: { label: 'Sector rezagado', tone: 'down' },
  neutral: { label: 'Sector neutral', tone: 'neutral' },
}

function WatchlistCard({ item }) {
  const tier = sectorTier(item.sector_rs_rank)
  const tierMeta = tier ? SECTOR_TIER_META[tier] : null

  return (
    <div className="watchlist-card">
      <div className="watchlist-card__header">
        <div>
          <span className="positions-table__ticker">{item.ticker}</span>
          <span className="watchlist-card__industry">{item.industry ?? item.sector}</span>
        </div>
        <span className="badge badge--buy">{HORIZON_LABELS[item.horizon]}</span>
      </div>
      <div className="watchlist-card__stats">
        <span>{formatCurrency(item.snapshot.price, item.snapshot.currency)}</span>
        <span className={item.snapshot.change_1d >= 0 ? 'delta-up' : 'delta-down'}>
          {formatPercent(item.snapshot.change_1d, { signed: true })} (1D)
        </span>
        {item.snapshot.rs_rating !== null && <span>RS {item.snapshot.rs_rating}</span>}
        <TrendBadge trend={item.snapshot.trend} />
      </div>
      {tierMeta && (
        <p className="watchlist-card__sector-tag">
          <span className={`sector-tier-badge sector-tier-badge--${tierMeta.tone}`}>
            {tierMeta.label} ({item.sector}, RS {item.sector_rs_rank})
          </span>
        </p>
      )}
      <ul className="watchlist-card__reasons">
        {item.reasons.map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>
    </div>
  )
}

function GeneralWatchlist({ region }) {
  const [horizon, setHorizon] = useState('')
  const [items, setItems] = useState([])
  const [computedAt, setComputedAt] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const body = await api.getWatchlist({ region, horizon: horizon || undefined })
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
  }, [region, horizon])

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      const body = await api.getWatchlist({ region, horizon: horizon || undefined, refresh: true })
      setItems(body.items)
      setComputedAt(body.computed_at)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setRefreshing(false)
    }
  }

  const aligned = items.filter((i) => sectorTier(i.sector_rs_rank) === 'strong')
  const rest = items.filter((i) => sectorTier(i.sector_rs_rank) !== 'strong')

  return (
    <div>
      <div className="filters-row">
        {HORIZONS.map((h) => (
          <button
            key={h.key}
            type="button"
            className={`timeframe-tab ${horizon === h.key ? 'timeframe-tab--active' : ''}`}
            onClick={() => setHorizon(h.key)}
          >
            {h.label}
          </button>
        ))}
      </div>

      {!loading && (
        <RefreshBar computedAt={computedAt} onRefresh={handleRefresh} refreshing={refreshing} />
      )}

      {loading ? (
        <p className="empty-state">Buscando candidatos…</p>
      ) : error ? (
        <div className="banner banner--error">{error}</div>
      ) : items.length === 0 ? (
        <p className="empty-state">Ningún activo cumple los criterios ahora mismo para este horizonte.</p>
      ) : (
        <div>
          {aligned.length > 0 && (
            <section className="watchlist-section">
              <h4 className="watchlist-section__title">
                Mejor alineadas - buen activo, además en un sector que está liderando ahora mismo
              </h4>
              <div className="watchlist-grid">
                {aligned.map((item) => (
                  <WatchlistCard key={`${item.ticker}-${item.horizon}`} item={item} />
                ))}
              </div>
            </section>
          )}

          <section className="watchlist-section">
            {aligned.length > 0 && <h4 className="watchlist-section__title">Resto de candidatos</h4>}
            <div className="watchlist-grid">
              {rest.map((item) => (
                <WatchlistCard key={`${item.ticker}-${item.horizon}`} item={item} />
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  )
}

const MODES = [
  { key: 'general', label: 'General' },
  { key: 'premium', label: 'Premium' },
]

function Watchlist({ onNavigateToTicker, region }) {
  const [mode, setMode] = useState('general')

  return (
    <div>
      <div className="sub-toggle" role="tablist" aria-label="Tipo de lista">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            role="tab"
            aria-selected={mode === m.key}
            className={`sub-toggle__item ${mode === m.key ? 'sub-toggle__item--active' : ''}`}
            onClick={() => setMode(m.key)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {mode === 'general' ? (
        <GeneralWatchlist region={region} />
      ) : (
        <PremiumWatchlist onNavigateToTicker={onNavigateToTicker} region={region} />
      )}
    </div>
  )
}

export default Watchlist
