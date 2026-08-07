import { useEffect, useState } from 'react'
import { api } from '../../api'
import TickerSnapshotTable from './TickerSnapshotTable'

const GROUPS = [
  { key: 'gainers', title: 'Mayores subidas (1D)', columns: ['price', 'change_1d', 'relative_volume'] },
  { key: 'losers', title: 'Mayores bajadas (1D)', columns: ['price', 'change_1d', 'relative_volume'] },
  { key: 'near_52w_high', title: 'Cerca del máximo 52 semanas', columns: ['price', 'dist_52w_high', 'change_1m'] },
  { key: 'near_52w_low', title: 'Cerca del mínimo 52 semanas', columns: ['price', 'dist_52w_low', 'change_1m'] },
  { key: 'high_volume', title: 'Volumen inusual', columns: ['price', 'change_1d', 'relative_volume'] },
  { key: 'oversold', title: 'Sobreventa (RSI bajo)', columns: ['price', 'rsi14', 'change_1m'] },
  { key: 'overbought', title: 'Sobrecompra (RSI alto)', columns: ['price', 'rsi14', 'change_1m'] },
  { key: 'golden_cross', title: 'Golden cross reciente (MA50/MA200)', columns: ['price', 'change_1m', 'trend'] },
  { key: 'death_cross', title: 'Death cross reciente (MA50/MA200)', columns: ['price', 'change_1m', 'trend'] },
  { key: 'rs_leaders', title: 'Líderes por RS Rating', columns: ['price', 'rs_rating', 'change_1m', 'stage'] },
  { key: 'strong_trend', title: 'Tendencia fuerte confirmada (ADX ≥ 25)', columns: ['price', 'adx14', 'change_1m'] },
]

function MarketMovers({ region }) {
  const [movers, setMovers] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        setMovers(await api.getMarketMovers({ region }))
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [region])

  if (loading) return <p className="empty-state">Cargando movers…</p>
  if (error) return <div className="banner banner--error">{error}</div>
  if (!movers) return null

  return (
    <div className="movers-grid">
      {GROUPS.map((group) => (
        <div key={group.key} className="movers-card">
          <p className="movers-card__title">{group.title}</p>
          <TickerSnapshotTable
            rows={movers[group.key] ?? []}
            columns={group.columns}
            showSector={false}
            emptyMessage="Sin candidatos ahora mismo."
          />
        </div>
      ))}
    </div>
  )
}

export default MarketMovers
